from graphembeddingnetwork import GraphEmbeddingNet
from graphembeddingnetwork import GraphPropLayer
import torch


def pairwise_euclidean_similarity(x, y):
    """Compute the pairwise Euclidean similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise euclidean similarity.
    """
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 0)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    """Compute the dot product similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise dot product similarity.
    """
    return torch.mm(x, torch.transpose(y, 1, 0))


def pairwise_cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise cosine similarity.
    """
    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), 1e-12)))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), 1e-12)))
    return torch.mm(x, torch.transpose(y, 1, 0))


PAIRWISE_SIMILARITY_FUNCTION = {
    'euclidean': pairwise_euclidean_similarity,
    'dotproduct': pairwise_dot_product_similarity,
    'cosine': pairwise_cosine_similarity,
}


def get_pairwise_similarity(name):
    """Get pairwise similarity metric by name.

    Args:
      name: string, name of the similarity metric, one of {dot-product, cosine,
        euclidean}.

    Returns:
      similarity: a (x, y) -> sim function.

    Raises:
      ValueError: if name is not supported.
    """
    if name not in PAIRWISE_SIMILARITY_FUNCTION:
        raise ValueError('Similarity metric name "%s" not supported.' % name)
    else:
        return PAIRWISE_SIMILARITY_FUNCTION[name]


def compute_cross_attention(x, y, sim):
    """Compute cross attention.

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i

    Args:
      x: NxD float tensor.
      y: MxD float tensor.
      sim: a (x, y) -> similarity function.

    Returns:
      attention_x: NxD float tensor.
      attention_y: NxD float tensor.
    """
    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


def batch_block_pair_attention(data,
                               block_idx,
                               n_blocks,
                               similarity='dotproduct'):
    """Compute batched attention between pairs of blocks.

    This function partitions the batch data into blocks according to block_idx.
    For each pair of blocks, x = data[block_idx == 2i], and
    y = data[block_idx == 2i+1], we compute

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

    and

    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i.

    Args:
      data: NxD float tensor.
      block_idx: N-dim int tensor.
      n_blocks: integer.
      similarity: a string, the similarity metric.

    Returns:
      attention_output: NxD float tensor, each x_i replaced by attention_x_i.

    Raises:
      ValueError: if n_blocks is not an integer or not a multiple of 2.
    """
    if not isinstance(n_blocks, int):
        raise ValueError('n_blocks (%s) has to be an integer.' % str(n_blocks))

    if n_blocks % 2 != 0:
        raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_blocks)

    sim = get_pairwise_similarity(similarity)

    results = []

    # This is probably better than doing boolean_mask for each i
    partitions = []
    for i in range(n_blocks):
        partitions.append(data[block_idx == i, :])

    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        attention_x, attention_y = compute_cross_attention(x, y, sim)
        results.append(attention_x)
        results.append(attention_y)
    results = torch.cat(results, dim=0)

    return results


class GraphPropMatchingLayer(GraphPropLayer):
    """A graph propagation layer that also does cross graph matching.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """
    # oyxy加
    def __init__(self,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 prop_type='embedding',
                 name='graph-net'):
        super(GraphPropMatchingLayer, self).__init__(
            node_state_dim,
            edge_state_dim,
            edge_hidden_sizes,
            node_hidden_sizes,
            edge_net_init_scale=edge_net_init_scale,
            node_update_type=node_update_type,
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            layer_norm=layer_norm,
            prop_type=prop_type,
            name=name
        )
        self.graph_idx_4edge = None

    # oyxy加
    def _compute_node_update(self,
                             node_states,
                             node_state_inputs,
                             node_features=None):
        """Compute node updates.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, the input node
            states.
          node_state_inputs: a list of tensors used to compute node updates.  Each
            element tensor should have shape [n_nodes, feat_dim], where feat_dim can
            be different.  These tensors will be concatenated along the feature
            dimension.
          node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.

        Returns:
          new_node_states: [n_nodes, node_state_dim] float tensor, the new node
            state tensor.

        Raises:
          ValueError: if node update type is not supported.
        """
        if self._node_update_type in ('mlp', 'residual'):
            node_state_inputs.append(node_states)
        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = torch.cat(node_state_inputs, dim=-1)

        if self._node_update_type == 'gru':
            node_state_inputs = torch.unsqueeze(node_state_inputs, 0)
            node_states = torch.unsqueeze(node_states, 0)
            _, new_node_states = self.GRU(node_state_inputs, node_states)
            new_node_states = torch.squeeze(new_node_states)
            return new_node_states
        else:
            mlp_output = self.MLP(node_state_inputs)
            if self._layer_norm:
                mlp_output = nn.self.layer_norm2(mlp_output)
            if self._node_update_type == 'mlp':
                return mlp_output
            elif self._node_update_type == 'residual':
                return node_states + mlp_output
            else:
                raise ValueError('Unknown node update type %s' % self._node_update_type)

    def forward(self,
                node_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                similarity='dotproduct',
                edge_features=None,
                node_features=None):
        """Run one propagation step with cross-graph matching.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          graph_idx: [n_onodes] int tensor, graph id for each node.
          n_graphs: integer, number of graphs in the batch.
          similarity: type of similarity to use for the cross graph attention.
          edge_features: if not None, should be [n_edges, edge_feat_dim] tensor,
            extra edge features.
          node_features: if not None, should be [n_nodes, node_feat_dim] tensor,
            extra node features.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.

        Raises:
          ValueError: if some options are not provided correctly.
        """
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features)

        # cross_graph_attention = batch_block_pair_attention(
        #     node_states, graph_idx, n_graphs, similarity=similarity)
        # attention_input = node_states - cross_graph_attention

        cross_graph_attention = batch_block_pair_attention(
            edge_features, self.graph_idx_4edge, n_graphs, similarity=similarity)
        attention_input = edge_features - cross_graph_attention

        # edge dim to node dim, note in this way edge_state_dim must equal to node_state_dim
        from segment import unsorted_segment_sum
        attention_input = unsorted_segment_sum(attention_input, to_idx, node_states.shape[0])

        return self._compute_node_update(node_states,
                                         [aggregated_messages, attention_input],
                                         node_features=node_features)


class GraphMatchingNet(GraphEmbeddingNet):
    """Graph matching net.

    This class uses graph matching layers instead of the simple graph prop layers.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """

    def __init__(self,
                 encoder,
                 aggregator,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 n_prop_layers,
                 share_prop_params=False,
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 layer_class=GraphPropLayer,
                 similarity='dotproduct',
                 prop_type='embedding'):
        super(GraphMatchingNet, self).__init__(
            encoder,
            aggregator,
            node_state_dim,
            edge_state_dim,
            edge_hidden_sizes,
            node_hidden_sizes,
            n_prop_layers,
            share_prop_params=share_prop_params,
            edge_net_init_scale=edge_net_init_scale,
            node_update_type=node_update_type,
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            layer_norm=layer_norm,
            layer_class=GraphPropMatchingLayer,
            prop_type=prop_type,
        )
        self._similarity = similarity

    def _apply_layer(self,
                     layer,
                     node_states,
                     from_idx,
                     to_idx,
                     graph_idx,
                     n_graphs,
                     edge_features):
        """Apply one layer on the given inputs."""
        return layer(node_states, from_idx, to_idx, graph_idx, n_graphs,
                     similarity=self._similarity, edge_features=edge_features)

    # oyxy加
    def add_edge_attributes(self, graph_idx_4edge):
        for layer in self._prop_layers:
            layer.graph_idx_4edge = graph_idx_4edge
