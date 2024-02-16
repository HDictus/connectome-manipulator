from bluepysnap import Circuit
import numpy as np
from pathlib import Path
import pytest
import pandas as pd
from utils import TEST_DATA_DIR
from connectome_manipulator import log
from connectome_manipulator.connectome_manipulation.converters import EdgeWriter
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


@pytest.fixture
def manipulation():
    m = Manipulation.get("conn_correlation")
    return m


def test_structural_placement(manipulation, tmp_path):

    tgt_ids, nodes, writer, struct_edges_table, constraints_path = _setup(tmp_path)
    manipulation(nodes, writer).apply(
        tgt_ids,
        None,
        struct_edges=struct_edges_table,
        sel_src={'mtype': 'L4_PC'},
        sel_dest={'mtype': 'L5_PC'},
        normalized_rates=constraints_path,
        **_default_models()
    )
    res = writer.to_pandas()
    src_nodes = nodes[0].ids({'mtype': 'L4_PC'})
    tgt_nodes = nodes[1].ids({'mtype': 'L5_PC'})
    rewired_edges = res[
        np.logical_and(
            np.isin(res['@source_node'], src_nodes),
            np.isin(res['@target_node'], tgt_nodes)
        )]
    assert _edges_are_subset(
        rewired_edges,
        struct_edges_table.drop(columns=['distance_soma', 'branch_order'])
    )


    
def test_only_creates_structurally_viable(manipulation, tmp_path):

    tgt_ids, nodes, writer, struct_edges_table, constraints_path = _setup(tmp_path)
    min_nsyn = 2
    edges_table = writer.to_pandas()
    edges_table = remove_small_connections(edges_table, min_nsyn)
    writer.from_pandas(edges_table)
    manipulation(nodes, writer).apply(
        tgt_ids,
        None,
        struct_edges=struct_edges_table,
        sel_src={'mtype': 'L4_PC'},
        sel_dest={'mtype': 'L5_PC'},
        normalized_rates=constraints_path,
        **_default_models()
    )
    res = writer.to_pandas()
    src_nodes = nodes[0].ids({'mtype': 'L4_PC'})
    tgt_nodes = nodes[1].ids({'mtype': 'L5_PC'})
    res = res[
        np.logical_and(
            np.isin(res['@source_node'], src_nodes),
            np.isin(res['@target_node'], tgt_nodes)
        )]

    napp = struct_edges_table.assign(napp=1).groupby(['@source_node', '@target_node'])['napp'].sum()
    pairs = res.set_index(['@source_node', '@target_node']).index
    for k, v in pairs:
        assert napp[(k, v)] >= min_nsyn


def remove_small_connections(edges, min_nsyn):
    nsyn = edges.assign(nsyn=1).groupby(['@source_node', '@target_node'])['nsyn'].sum()
    edges = edges.set_index(['@source_node', '@target_node'])[nsyn >= min_nsyn]
    return edges.reset_index()


def _setup(tmp_path):
    log.setup_logging()
    c = Circuit(Path(TEST_DATA_DIR, "circuit_sonata.json"))

    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)
    edges_table = edges_table.sort_values('@target_node').reset_index(drop=True)

    struct_config = Path(TEST_DATA_DIR, "circuit_sonata_struct.json")
    struct_edges_table = _load_struct_edges(struct_config)
    writer = EdgeWriter(None, edges_table.copy())
    activity = _create_random_normalized_activity(np.union1d(nodes[0].ids(), nodes[1].ids()))
    constraints_path = tmp_path / 'activity.feather'
    activity.reset_index().to_feather(constraints_path)
    return tgt_ids, nodes, writer, struct_edges_table, constraints_path 


def _create_random_normalized_activity(ids):
    activity = pd.DataFrame(
        [{'gid': id_,
          'other_col': other,
          'rate': np.random.uniform()}
         for other in range(100) for id_ in ids])
    centered = activity.set_index('gid') / activity.groupby('gid').mean()
    return centered / activity.groupby('gid').std()

         
def _load_struct_edges(struct_config):
    c = Circuit(struct_config)
    edges = c.edges[c.edges.population_names[0]]
    return edges.afferent_edges(edges.target.ids(), properties=edges.property_names)

def _default_models():
    return {
        "delay_model": {
            "model": "LinDelayModel",
            "delay_mean_coeff_a": 0.1, # base delay for release of transmitter
            "delay_mean_coeff_b": 1/300, # delay per um3 axon
            "delay_std": 0,
            "delay_min": 0.1

        }
    }



def _create_random_rcorr(struct_edges_table):
    struct_constraint = struct_edges_table.assign(n_appositions=1).groupby(
        ['@source_node', '@target_node']
    )['n_appositions'].sum()
    return struct_constraint.reset_index().assign(
        r=np.random.uniform(-1, 1, size=len(struct_constraint))
    )

def _edges_are_subset(result, superset):
    super = superset.set_index(list(superset.columns)).index
    sub = result.set_index((list(superset.columns))).index
    return len(sub.difference(super)) == 0


def test_uses_distance_soma_for_delay(manipulation, tmp_path):

    tgt_ids, nodes, writer, struct_edges_table, constraints_path = _setup(tmp_path)
    manipulation(nodes, writer).apply(
        tgt_ids,
        None,
        struct_edges=struct_edges_table,
        sel_src={'mtype': 'L4_PC'},
        sel_dest={'mtype': 'L5_PC'},
        normalized_rates=constraints_path,
        delay_model={
            "model": "LinDelayModel",
            "delay_mean_coeff_a": 0.1, # base delay for release of transmitter
            "delay_mean_coeff_b": 1/300, # delay per um3 axon
            "delay_std": 0,
            "delay_min": 0.1
        },
    )
    res = writer.to_pandas()
    src_nodes = nodes[0].ids({'mtype': 'L4_PC'})
    tgt_nodes = nodes[1].ids({'mtype': 'L5_PC'})
    rewired_edges = res[
        np.logical_and(
            np.isin(res['@source_node'], src_nodes),
            np.isin(res['@target_node'], tgt_nodes)
        )]
    structural_properties = [c for c in struct_edges_table if c in rewired_edges]
    struct_rewired = struct_edges_table.set_index(structural_properties).loc[
        rewired_edges.set_index(structural_properties).index]
    assert np.allclose(
        rewired_edges['delay'],
        struct_rewired['distance_soma'] / 300 + 0.1
    )

def test_not_all_rcorr_in_struct(manipulation, tmp_path):
    tgt_ids, nodes, writer, struct_edges_table, constraints_path = _setup(tmp_path)
    tgt_ids = [9]
    edges_table = writer.to_pandas()
    edges_table = edges_table[
        np.isin(edges_table['@target_node'], tgt_ids)
    ]
    struct_edges_table = struct_edges_table[
        np.isin(struct_edges_table['@target_node'], tgt_ids)
    ]
    
    manipulation(nodes, writer).apply(
        tgt_ids,
        None,
        struct_edges=struct_edges_table,
        sel_src={'mtype': 'L4_PC'},
        sel_dest={'mtype': 'L5_PC'},
        normalized_rates=constraints_path,
        **_default_models()
    )
    res = writer.to_pandas()
    src_nodes = nodes[0].ids({'mtype': 'L4_PC'})
    tgt_nodes = nodes[1].ids({'mtype': 'L5_PC'})
    rewired_edges = res[
        np.logical_and(
            np.isin(res['@source_node'], src_nodes),
            np.isin(res['@target_node'], tgt_nodes)
        )]
    assert _edges_are_subset(
        rewired_edges,
        struct_edges_table.drop(columns=['distance_soma', 'branch_order'])
    )


def test_no_conns(manipulation, tmp_path):
    tgt_ids, nodes, writer, struct_edges_table, constraints_path = _setup(tmp_path)
    edges_table = writer.to_pandas()
    manipulation(nodes, writer).apply(
        tgt_ids,
        None,
        struct_edges=struct_edges_table,
        sel_src={'mtype': 'bababab'},
        sel_dest={'mtype': 'bababab'},
        normalized_rates=constraints_path,
        **_default_models()
    )
    res = writer.to_pandas()
    pd.testing.assert_frame_equal(
        edges_table,
        res)
