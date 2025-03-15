
'''
sample:
    song_model.sample(condition)
        song_model.sample_one([], condition)
            condition = song_model.forward([],condition)

            frame_model.sample(condition)
                frame_model.sample_one([], condition)
                    condition = frame_model.forward([], condition)
                    pitch_model.sample(condition)
                        pitch_model.classifier(condition)
                pitch_node_compact = pitch_model.get_compact(pitch_node)
                frame_model.sample_one([pitch_node_compact], condition)
                    condition = frame_model.forward([pitch_node_compact], condition)
                    velocity_model.sample(condition)
                        velocity_model.classifier(condition)
                return [frame_node([pitch_node, velocity_node])]

            frame_node_compact = frame_model.get_compact()
                pitch_node_compact = pitch_model.get_compact(pitch_node)
                velocity_node_compact = velocity_model.get_compact(velocity_node)
                return self.get_compact([pitch_node_compact, velocity_node_compact])

            condition = song_model.forward([frame_node_compact], condition)
            frame_model.sample(condition)
            ...

calc_loss:
    song_model.calc_loss(condition)
        embed_sequence = []
        frame_model.get_compact(children[0])
            pitch_model.get_compact(children[0])
            velocity_model.get_compact(children[1])
            return self.get_compact([...])
        embed_sequence.append(frame_node_compact)
        ...
        condition = song_model.forward(embed_sequence)
            frame_model.calc_loss(condition)
                loss_pitch = pitch_model.calc_loss(condition)
                loss_velocity = velocity_model.calc_loss(condition)
                return loss_pitch + loss_velocity
            return loss
        return loss
'''

from collections import defaultdict
import torch
import torch.nn as nn
from typing import Any, TypeVar, Generic, Sequence, Callable, TypedDict
from abc import ABC, abstractmethod
from torch.nn import functional as F

class Node:
    def __init__(self, children: Sequence['Node']):
        super().__init__()
        self.children = children

T = TypeVar('T', bound='Node')
class NodeModel(nn.Module, Generic[T], ABC):
    '''
    A model that can sample a node from a given condition.
    '''
    def __init__(self, node_type: type[T]):
        super().__init__()
        self.node_type = node_type

    @abstractmethod
    def sample(self, condition: Any, get_compact: Callable[[T], torch.Tensor]) -> T:
        raise NotImplementedError
    
    @abstractmethod
    def calc_loss(self, gt_node: list[T], condition: Any, get_compact: Callable[[list[T]], Any]) -> torch.Tensor:
        '''
        gt_node: a batch of nodes
        condition: b, c
        get_compact: a batch of nodes -> a batch of compacts (b, ...)
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_compact(self, children: torch.Tensor) -> torch.Tensor:
        '''
        children: a batch of compacts of children (b, l, ...)
        return: a batch of compacts of self (b, ...)
        '''
        raise NotImplementedError
    
    def to(self, device: torch.device):
        self.device = device
        super().to(device)
    
class AutoRegressiveNodeModel(NodeModel[T]):
    '''
    A node model that samples its children in an auto-regressive manner.
    '''
    def __init__(self, node_type: type[T]):
        super().__init__(node_type)


    def sample(self, condition: Any, get_compact: Callable[[T], Any]) -> T:
        children = []
        children_compact = []
        while True:
            child = self._sample_one(children, condition)
            if child is None:
                break
            children.append(child)
            children_compact.append(get_compact(child))
        new_node = self.node_type(children_compact)
        return new_node

    def calc_loss(self, gt_nodes: list[T], condition: Any, get_compact: Callable[[Node], Any], get_model: Callable[[type[Node]], NodeModel]) -> torch.Tensor:
        children_grouped_by_type = defaultdict(list)
        batch_idx_grouped_by_type = defaultdict(list)
        for batch_idx, node in enumerate(gt_nodes):
            for child in node.children:
                children_grouped_by_type[type(child)].append(get_compact(child))
                batch_idx_grouped_by_type[type(child)].append(batch_idx)

        # call calc_loss for each child type
        loss = torch.zeros(1, device=self.device)

        for child_type, children in children_grouped_by_type.items():
            condition_for_children = torch.gather(condition, 0, torch.tensor(batch_idx_grouped_by_type[child_type], device=self.device))
            child_model = get_model(child_type)
            child_loss = child_model(condition_for_children, children)
            loss += child_loss
        return loss
    
    @abstractmethod
    def _sample_one(self, prev_children: list, condition: Any) -> T|None:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, children_compact: list, condition: Any) -> torch.Tensor:
        '''
        Compute the condition to pass down as the condition of children.
        '''
        raise NotImplementedError

def group_children_by_type(nodes: list[Node]) -> tuple[dict[type, list[Node]], dict[type, list[int]]]:
    children_grouped = defaultdict(list)
    batch_idx_grouped = defaultdict(list)
    for batch_idx, node in enumerate(nodes):
        for child in node.children:
            children_grouped[type(child)].append(child)
            batch_idx_grouped[type(child)].append(batch_idx)
    return children_grouped, batch_idx_grouped
    
class CompactAndChildrenCompact(TypedDict):
    compact: torch.Tensor
    children_compact: torch.Tensor

class TreeGenerator(nn.Module):
    '''
    A node model that samples a tree from a given condition.
    '''
    def __init__(self, node_type_model_map: dict[type[Node], NodeModel], root_node_model: NodeModel):
        super().__init__()
        self.node_type_model_map: dict[type[Node], NodeModel] = node_type_model_map
        self.root_node_model: NodeModel = root_node_model

    def update_node_type(self, node_type: type[Node], model: NodeModel):
        self.node_type_model_map[node_type] = model

    def get_compact(self, nodes: list[Node]) -> torch.Tensor:
        model = self.node_type_model_map[type(nodes[0])]
        return model.get_compact(nodes)
    
    def get_all_compacts(self, root_batch: list[Node]) -> dict[Node, CompactAndChildrenCompact]:
        '''
        Recursively get all compacts for each node in the tree.
        All nodes in root_batch must have the same type.
        '''
        result_compacts: dict[Node, CompactAndChildrenCompact] = {}
        def get_compacts_recursive(node_type: type[Node], node_batch: list[Node]):
            children, batch_idx = group_children_by_type(node_batch)
            for child_type, children_of_type in children.items():
                get_compacts_recursive(child_type, children_of_type)

            # now compacts has all compacts for children of each node in node_batch.
            
            # stack length dim

            children_compact_batch = []
            for node in node_batch:
                # stack the compacts of the children
                children_compact = [result_compacts[child]['compact'] for child in node.children]
                children_compact = torch.stack(children_compact, dim=1) # (l, ...)
                children_compact_batch.append(children_compact)

            # pad length dim and stack batch dim
            max_length = max(len(compact) for compact in children_compact_batch)
            for i, compact in enumerate(children_compact_batch):
                compact = torch.cat([compact, torch.zeros(max_length - len(compact), *compact.shape[1:], device=compact.device)])
                children_compact_batch[i] = compact
    
            children_compact_batch = torch.stack(children_compact_batch, dim=0) # (b, l, ...)

            # forward get_compact for the current node (in a batch)
            compacts = self.node_type_model_map[node_type].get_compact(children_compact_batch) # (b, ...)

            for node, compact in zip(node_batch, compacts):
                result_compacts[node] = {'compact': compact, 'children_compact': children_compact_batch}
        
        get_compacts_recursive(type(root_batch[0]), root_batch)
        return result_compacts
    
    def sample(self, condition: Any = None, node_model: NodeModel|None=None) -> Node:
        if node_model is None:
            assert self.root_node_model is not None, "Root node model is not set so you must pass node_model argument to sample."
            node_model = self.root_node_model
            
        return node_model.sample(condition, self.get_compact)
    
    def calc_loss(self, gt_node: list[Node], condition: Any = None, node_model: NodeModel|None=None) -> torch.Tensor:
        if node_model is None:
            assert self.root_node_model is not None, "Root node model is not set so you must pass node_model argument to calc_loss."
            node_model = self.root_node_model
        model = self.node_type_model_map[type(gt_node[0])]
        compacts = self.get_all_compacts(gt_node)
        return self._calc_loss(gt_node, model, condition, compacts)
        
    def _calc_loss(self, gt_node: list[Node], node_model: NodeModel, condition: Any, compacts: dict[Node, CompactAndChildrenCompact]) -> torch.Tensor:
        condition_for_children = node_model(compacts[gt_node]['children_compact'], condition)
        losses = []
        for node, compact in zip(gt_node, compacts):
            loss = node_model(condition_for_children, compact['compact'])
            losses.append(loss)
        return torch.stack(losses, dim=0).mean()

    def forward(self, node: Node, condition: Any) -> torch.Tensor:
        model = self.node_type_model_map[type(node)]
        return model.forward(condition, node)

if __name__ == '__main__':
    class PitchNode(Node):
        def __init__(self, value: int):
            super().__init__([])
            self.value = value

    class VelocityNode(Node):
        def __init__(self, value: int):
            super().__init__([])
            self.value = value

    class NoteNode(Node):
        def __init__(self, pitch: PitchNode, velocity: VelocityNode):
            super().__init__([pitch, velocity])
            self.pitch = pitch
            self.velocity = velocity

    class NoteSequenceNode(Node):
        def __init__(self, notes: Sequence[NoteNode]):  
            super().__init__(notes)
            self.notes = notes

    class PitchNodeModel(AutoRegressiveNodeModel[PitchNode]):
        def __init__(self, input_dim: int):
            super().__init__(PitchNode)

            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 128),
            )

            self.crit = nn.CrossEntropyLoss()

        def forward(self, condition: Any, gt_node: list[PitchNode]) -> torch.Tensor:
            '''
            condition: b, f
            gt_node: b, n
            '''
            gt = torch.tensor([node.value for node in gt_node], dtype=torch.long, device=self.device)
            logits = self.classifier(condition)
            return self.crit(logits, gt)

        def _sample_one(self, prev_children: list, condition: Any) -> PitchNode|None:
            logits = self.classifier(condition)
            probs = F.softmax(logits, dim=-1)
            return PitchNode(int(torch.multinomial(probs, num_samples=1).item()))
    