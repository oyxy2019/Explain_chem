def get_default_config():
    """The default configs."""
    model_type = 'matching'
    # Set to `embedding` to use the graph embedding net.
    node_state_dim = 64
    edge_state_dim = 64
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_state_dim=edge_state_dim,
        edge_hidden_sizes=[edge_state_dim * 2, edge_state_dim * 2, edge_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2, node_state_dim * 2, node_state_dim * 2],
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=False,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here. gru
        edge_update_type='gru',     # {gru, mlp, residual}
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=False,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
        # set to `embedding` to use the graph embedding net.
        prop_type=model_type)
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config['similarity'] = 'dotproduct'  # {dotproduct, euclidean, cosine}
    return dict(
        node_state_dim=node_state_dim,
        edge_state_dim=edge_state_dim,
        graph_rep_dim=graph_rep_dim,
        encoder=dict(
            node_hidden_sizes=[node_state_dim*2, node_state_dim*2, node_state_dim],
            edge_hidden_sizes=[edge_state_dim*2, edge_state_dim*2, edge_state_dim]),
        aggregator=dict(
            edge_hidden_sizes=[graph_rep_dim, graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim, graph_rep_dim],
            input_size=[edge_state_dim],
            gated=True,
            aggregation_type='sum'),    # {sum, max, mean, sqrt_n}
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        model_type=model_type,
        training=dict(
            num_epoch=200,
            patience=50,
            learning_rate=1e-3,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=1.0,),
        seed=8,
    )


def change_config(node_state_dim, edge_state_dim, graph_rep_dim):
    """The default configs."""
    model_type = 'matching'
    # Set to `embedding` to use the graph embedding net.
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_state_dim=edge_state_dim,
        edge_hidden_sizes=[edge_state_dim * 2, edge_state_dim * 2, edge_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2, node_state_dim * 2, node_state_dim * 2],
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=False,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here. gru
        edge_update_type='gru',     # {gru, mlp, residual}
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=False,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
        # set to `embedding` to use the graph embedding net.
        prop_type=model_type)
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config['similarity'] = 'dotproduct'  # {dotproduct, euclidean, cosine}
    return dict(
        node_state_dim=node_state_dim,
        edge_state_dim=edge_state_dim,
        graph_rep_dim=graph_rep_dim,
        encoder=dict(
            node_hidden_sizes=[node_state_dim*2, node_state_dim*2, node_state_dim],
            edge_hidden_sizes=[edge_state_dim*2, edge_state_dim*2, edge_state_dim]),
        aggregator=dict(
            edge_hidden_sizes=[graph_rep_dim, graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim, graph_rep_dim],
            input_size=[edge_state_dim],
            gated=True,
            aggregation_type='sum'),    # {sum, max, mean, sqrt_n}
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        model_type=model_type,
        training=dict(
            num_epoch=200,
            patience=50,
            learning_rate=1e-3,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=1.0,),
        seed=8,
    )
