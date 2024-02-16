import numpy as np
import pandas as pd
from connectome_manipulator import log, profiler
from connectome_manipulator.model_building.model_types import AbstractModel
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


class ResponseCorrelationRewiring(Manipulation):
    

    def apply(self,
              split_ids,
              syn_class,
              struct_edges,
              rcorr_file,
              delay_model,
              pathway_specs=None,
              sel_src=None,
              sel_dest=None
            ):

        log.debug("Loading response correlation")
        rcorr = pd.read_feather(rcorr_file)
        log.debug("Restricting edges to relevant pathway.")
        all_edges = self.writer.to_pandas()
        src_nodes = self.nodes[0].ids(sel_src)
        tgt_nodes = np.intersect1d(
            self.nodes[0].ids(sel_dest),
            split_ids
        )
        in_pathway = np.logical_and(
            np.isin(all_edges['@source_node'], src_nodes),
            np.isin(all_edges['@target_node'], tgt_nodes)
        )
        if in_pathway.sum() == 0:
            return
        prev_edges = all_edges[in_pathway]

        rcorr = rcorr[
            np.logical_and(
                np.isin(rcorr['@source_node'], src_nodes),
                np.isin(rcorr['@target_node'], tgt_nodes)
            )
        ]

        rcorr = rcorr.set_index(['@source_node', '@target_node'])
        n_appositions = rcorr['n_appositions']
        rcorr = rcorr['r']

        log.debug("Grouping structural and functional connections")

        new_edges = _rewire_based_on_rcorr(prev_edges, rcorr, n_appositions, delay_model, struct_edges)

        all_edges = pd.concat([all_edges[~in_pathway], new_edges])
        all_edges = all_edges.sort_values(['@target_node', '@source_node']).reset_index(drop=True)
        self.writer.from_pandas(all_edges)        



def _rewire_based_on_rcorr(prev_edges, rcorr, n_appositions, delay_model, struct_edges):    
    previous_conns = _collapse_connections(prev_edges)
    rcorr_by_appositions = rcorr.groupby(n_appositions)
    log.debug("Reassigning connections")
    out_conns = []
    for n_app, conns in previous_conns.groupby(n_appositions):
        if not all(conns['nsyn'] <= n_app):
            print(conns)
            import pdb; pdb.set_trace()
            raise ValueError("structural impossibility")

        connections = _assign_connections(conns, rcorr_by_appositions.get_group(n_app))
        out_conns.append(connections)

    if len(out_conns) == 0:
        import pdb; pdb.set_trace()

    new_connections = pd.concat(out_conns)

    log.debug("Reassigning synapse properties")
    new_edges = _retrieve_synapse_properties(
        prev_edges,
        new_connections[['previous_source', 'previous_target']],
        new_connections[['@source_node', '@target_node']]
    )

    log.debug("Placing synapses")
    _place_synapses_from_structural(new_edges, struct_edges)
    log.debug("Assigning delays")
    _assign_delays_from_model(delay_model, new_edges)

    log.debug("Rewired.")
    
    return new_edges


def _assign_connections(connections, based_on):
    connections = connections.reset_index().sort_values('conductance')
    connections[['previous_source', 'previous_target']] = connections[
        ['@source_node', '@target_node']
    ].copy()
    based_on = based_on.sort_values()
    to_connect = based_on.iloc[:len(connections)].reset_index()
    connections[['@source_node', '@target_node']] =\
        to_connect[['@source_node', '@target_node']].values
    return connections


def _retrieve_synapse_properties(prev_edges, prev_pairs, new_pairs):
    indexed_by_prev_pair = prev_edges.set_index(['@source_node', '@target_node']).sort_index()
    synapses = []
    for (prev_src, prev_tgt), (new_src, new_tgt) in zip(prev_pairs.values, new_pairs.values):
        connection_synapses = indexed_by_prev_pair.loc[(prev_src, prev_tgt)].copy()
        if isinstance(connection_synapses, pd.Series):
            connection_synapses = pd.DataFrame({**connection_synapses}, index=[0])
        connection_synapses.reset_index(inplace=True, drop=True)
        connection_synapses['@source_node'] = int(new_src)
        connection_synapses['@target_node'] = int(new_tgt)
        synapses.append(connection_synapses)

    if len(synapses) == 0:
        import pdb; pdb.set_trace()
    new_edges = pd.concat(synapses).reset_index(drop=True)
    return new_edges


def _place_synapses_from_structural(new_edges, structural_edges):
    structural_edges = structural_edges.set_index(['@source_node', '@target_node']).sort_index()
    if len(new_edges) == 0:
        return pd.DataFrame([], columns=structural_edges.columns)
    for (pre, post), conn in new_edges.groupby(['@source_node', '@target_node']):
        try:
            candidates = structural_edges.loc[(pre, post)]
        except KeyError:
            import pdb; pdb.set_trace()
        replace = False
        if len(conn) > len(candidates):
            raise ValueError("more synapses than synapse sites.")
            log.warning(f"Not enough unique synapse sites for connection ({pre}, {post})"
                        f"with {len(conn)} synapses {len(candidates)} sites. reusing synapses")
            replace = True

        values = candidates.iloc[np.random.choice(range(len(candidates)), size=len(conn), replace=replace)]
        if isinstance(values, pd.Series):
            import pdb; pdb.set_trace()
        new_edges.loc[conn.index, values.columns] = values.values
            

def _assign_delays_from_model(delay_model, new_edges):
    if len(new_edges) == 0:
        return new_edges
    new_edges['delay'] = AbstractModel.init_model(delay_model).apply(
        distance=new_edges['distance_soma']
    )


def _collapse_connections(df):
    grouped = df.reset_index().assign(nsyn=1).groupby(['@source_node', '@target_node'])
    columns = ['nsyn']
    if 'conductance' in df.columns:
        columns += ['conductance']
    return grouped[columns].sum().sort_index()
