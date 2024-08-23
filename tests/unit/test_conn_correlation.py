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

def test_rcorr_rank_equals_connductance_rank_within_tgid(manipulation, tmp_path):
    tgt_ids, nodes, writer, struct_edges_table, constraints_path = _setup(tmp_path)
    edges_table = writer.to_pandas()

    # control structural factor to isolate correlation
    napp = 3 
    edges_table = remove_large_connections(edges_table, napp)
    struct_edges_table = set_apposition_count(struct_edges_table, napp)
    
    writer.from_pandas(edges_table)
    manipulation(nodes, writer).apply(
        tgt_ids,
        None,
        struct_edges=struct_edges_table,
        sel_src={},
        sel_dest={},
        normalized_rates=constraints_path,
        **_default_models()
    )
    activity = pd.read_feather(constraints_path)
    rcorr = activity.pivot_table(index='other_col', columns='gid', values='rate').corr().melt(ignore_index=False).rename(columns={'gid': '@target_node'}).reset_index().rename(columns={'gid': '@source_node'}).set_index(['@source_node', '@target_node'])['value']
    napp = struct_edges_table.assign(napp=1).groupby(['@source_node', '@target_node'])['napp'].sum()
    rcorr = rcorr[napp.index]
    res = writer.to_pandas()
    conductance = res.groupby(['@source_node', '@target_node'])['conductance'].sum()

    
    
    combined = pd.DataFrame({'conductance': conductance, 'rcorr': rcorr}).dropna()
    combined['tgid'] = combined.index.get_level_values(1)
    # we sort by both below to ensure that with reciprocal connections the order is still ok
    #  if two pairs have the same rcorr.

    assert all(
        combined.sort_values(['tgid', 'conductance']).index ==
        combined.sort_values(['tgid', 'rcorr', 'conductance']).index)

def set_apposition_count(struct_edges, napp):
    apps = []
    for (src, tgt), appositions in struct_edges.groupby(['@target_node', '@source_node']):
        if len(appositions) < napp:
            factor = napp / len(appositions) 
            appositions = pd.concat([appositions] * int(np.ceil(factor)), axis=0)
        if len(appositions) > napp:
            appositions = appositions.iloc[:napp]
        apps.append(appositions)
    return pd.concat(apps, axis=0).reset_index(drop=True)
        

def test_one_struct_one_conn(manipulation, tmp_path):
    # TODO: testing this on smaller util methods would be clearer
    tgt_ids, nodes, writer, struct_edges_table, constraints_path = _setup(tmp_path)
    edges_table = writer.to_pandas()
    edges_table = edges_table.iloc[[0]]
    struct_edges_table = struct_edges_table.set_index(['@source_node', '@target_node']).loc[
        tuple(edges_table[['@source_node', '@target_node']].values[0])].iloc[[0]].reset_index()
    writer.from_pandas(edges_table)
    manipulation(nodes, writer).apply(
        tgt_ids,
        None,
        struct_edges=struct_edges_table,
        sel_src={},
        sel_dest={},
        normalized_rates=constraints_path,
        **_default_models()
    )
    res = writer.to_pandas()
    assert all(
        res[['@source_node', '@target_node']] ==
        struct_edges_table[['@source_node', '@target_node']]
    )

    
@pytest.mark.parametrize('exc_num', range(20))
def test_only_creates_structurally_viable(manipulation, tmp_path, exc_num):

    tgt_ids, nodes, writer, struct_edges_table, constraints_path = _setup(tmp_path)
    min_nsyn = 2
    edges_table = writer.to_pandas()
    edges_table = remove_small_connections(edges_table, min_nsyn)
    assert(len(edges_table) > 0)
    writer.from_pandas(edges_table)
    manipulation(nodes, writer).apply(
        tgt_ids,
        None,
        struct_edges=struct_edges_table,
        sel_src={},
        sel_dest={},
        normalized_rates=constraints_path,
        **_default_models()
    )
    res = writer.to_pandas()
    napp = struct_edges_table.assign(napp=1).groupby(['@source_node', '@target_node'])['napp'].sum()
    nsyn = res.assign(nsyn=1).groupby(['@source_node', '@target_node'])['nsyn'].sum()
    assert all(napp[nsyn.index] >= nsyn)


def remove_small_connections(edges, min_nsyn):
    nsyn = edges.assign(nsyn=1).groupby(['@source_node', '@target_node'])['nsyn'].sum()
    edges = edges.set_index(['@source_node', '@target_node'])[nsyn >= min_nsyn]
    return edges.reset_index()


def remove_large_connections(edges, max_nsyn):
    nsyn = edges.assign(nsyn=1).groupby(['@source_node', '@target_node'])['nsyn'].sum()
    edges = edges.set_index(['@source_node', '@target_node'])[nsyn <= max_nsyn]
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
    activity.to_feather(constraints_path)
    return tgt_ids, nodes, writer, struct_edges_table, constraints_path 


def _create_random_normalized_activity(ids):
    activity = pd.DataFrame(
        [{'gid': id_,
          'other_col': other,
          'rate': np.random.uniform()}
         for other in range(100) for id_ in ids])
    centered = activity.set_index(['gid', 'other_col'])
    centered['rate'] -= centered.groupby('gid')['rate'].mean()
    centered['rate'] /= centered.groupby('gid')['rate'].std()
    assert all(centered.groupby('gid').std() == 1)
    return centered.reset_index()

         
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

# TODO: we can probably address this flakiness if we use finer-grained testing
#   or mocking 
#   that lets us set the rcorr direclty
@pytest.mark.parametrize('exc_num', range(20))
def test_total_in_conductance_conserved(manipulation, tmp_path, exc_num):

    tgt_ids, nodes, writer, struct_edges_table, constraints_path = _setup(tmp_path)
    edges_table = writer.to_pandas()
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
    assert np.allclose(
        edges_table.groupby('@target_node')['conductance'].sum(),
        res.groupby('@target_node')['conductance'].sum())
