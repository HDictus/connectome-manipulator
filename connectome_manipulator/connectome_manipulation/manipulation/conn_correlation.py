import numpy as np
import pandas as pd
from tqdm import tqdm
from connectome_manipulator import log, profiler
from connectome_manipulator.model_building.model_types import AbstractModel
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


class ResponseCorrelationRewiring(Manipulation):
    

    def apply(self,
              split_ids,
              syn_class,
              struct_edges,
              normalized_rates,
              delay_model=None,
              pathway_specs=None,
              sel_src=None,
              sel_dest=None,
              structural_constraint='min',
            ):

        log.debug("Loading response correlation")
        activity = _load_activity(normalized_rates)

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
        
        viable_connections = _determine_structurally_viable(struct_edges, prev_edges)
        rcorr = _calculate_response_correlation(activity, viable_connections.index)

        log.debug("Grouping structural and functional connections")

        new_edges = _rewire_based_on_rcorr(prev_edges, rcorr, viable_connections, delay_model, struct_edges, structural_constraint=structural_constraint)

        all_edges = pd.concat([all_edges[~in_pathway], new_edges])
        all_edges = all_edges.sort_values(['@target_node', '@source_node']).reset_index(drop=True)
        self.writer.from_pandas(all_edges)        

def _load_activity(normalized_rates):
    # TODO: load as pivot table
    activity = pd.read_feather(normalized_rates)

    other_cols = [col for col in activity if col not in ['gid', 'rate']]
    
    activity = activity.set_index(['gid'] + other_cols, drop=True)['rate'].sort_index()
    return activity

def _determine_structurally_viable(struct_edges, prev_edges):
    n_appositions = _collapse_connections(struct_edges)['nsyn']
    n_syns = _collapse_connections(prev_edges)['nsyn']
    return n_appositions[n_appositions >= n_syns.min()]


def _calculate_response_correlation(activity, pairs):
    """Activity must be centered and normalized."""
    print("rcorr")
    for _ in range(1):
        other_columns = [c for c in activity.reset_index().columns if c not in ['gid', 'rate']]
        pivoted = activity.fillna(0).reset_index().pivot_table(
            values='rate',
            index='gid',
            columns=other_columns)
        pre, post = pairs.get_level_values(0), pairs.get_level_values(1)
        
        pre_activity = pivoted.loc[pre].values
        post_activity = pivoted.loc[post].values
        rcorr =  pd.DataFrame({
            'r': (pre_activity * post_activity).mean(axis=1),
            '@source_node': pre,
            '@target_node': post})
        return rcorr.set_index(['@source_node', '@target_node'])['r']


def _rewire_based_on_rcorr(prev_edges, rcorr, n_appositions, delay_model, struct_edges, structural_constraint='min'):
    assert structural_constraint in ('min', 'max')
    previous_conns = _collapse_connections(prev_edges).sort_values('nsyn', ascending=False)
    # rcorr_by_appositions = rcorr.groupby(n_appositions)
    log.debug("Reassigning connections")
    based_on = rcorr #* n_appositions

    if structural_constraint == 'max':
        groups = pd.DataFrame({'r': rcorr}).sort_values(
            ['@target_node', 'r']).groupby('@target_node')
        rank_in_tgid = pd.DataFrame(
            {'pct': (groups.cumcount() / (groups['r'].count()-1)),
             'r': rcorr}).sort_values(['pct', 'r'], ascending=False)

        rank_by_napp = rank_in_tgid['r'].groupby(n_appositions)
        out_conns = []
        for napp, conns in previous_conns.groupby(n_appositions):

            out_conns.append(
                _assign_connections(conns, rank_by_napp.get_group(napp), n_appositions, 'max')
            )
    else:
        gb = '@target_node'

        groups = pd.DataFrame(
            {'napp': n_appositions},
            index=n_appositions.index
        )
        grouper = pd.concat([
            pd.Series(i, index=group.index)
            for i, (_, group) in enumerate(groups.groupby(gb))], axis=0)

        bo_by_tgid = based_on.sort_values(ascending=False).groupby(grouper)
        out_conns = []
        for group, conns in previous_conns.groupby(grouper):
            out_conns.append(_assign_connections(conns, bo_by_tgid.get_group(group), n_appositions, structural_constraint=structural_constraint))
    new_connections = pd.concat(out_conns)

    # for n_app, conns in tqdm(previous_conns.groupby(n_appositions)):
    #     if not all(conns['nsyn'] <= n_app):
    #         print(conns)
    #         raise ValueError("structural impossibility")

    #     connections = _assign_connections(conns, rcorr_by_appositions.get_group(n_app))
    #     out_conns.append(connections)
    # print("reassigned connections")
    # new_connections = pd.concat(out_conns)

    log.debug("Reassigning synapse properties")
    new_edges = _retrieve_synapse_properties(
        prev_edges,
        new_connections[['previous_source', 'previous_target']],
        new_connections[['@source_node', '@target_node']]
    )
    new_edges = _shift_synapses(new_edges, rcorr)
    log.debug("Placing synapses")
    new_edges = _place_synapses_from_structural(new_edges, struct_edges, new_connections)
    log.debug("Assigning delays")
    _assign_delays_from_model(delay_model, new_edges)

    log.debug("Rewired.")
    
    return new_edges


def _assign_connections(connections, based_on, n_appositions, structural_constraint='min'):
    num_conns = len(connections)
    connections = connections.reset_index()#.sort_values(['nsyn'], ascending=False)
    connections[['previous_source', 'previous_target']] = connections[
        ['@source_node', '@target_node']
    ].copy()
    #based_on = based_on.sort_values(ascending=False)

    if structural_constraint == 'max':
        # import pdb; pdb.set_trace()
        to_connect = based_on.reset_index()[['@source_node', '@target_node']].values[:len(connections)]
        connections[['@source_node', '@target_node']] = to_connect
    else:
        matched = []
        possible_connections = connections['nsyn']

        for (pre, post), value in based_on.items():
            if len(possible_connections) == 0:
                break
            n_app = n_appositions[(pre, post)]
            i = 0
            for j, nsyn in possible_connections.items():
                if nsyn <= n_app:
                    matched.append({'index': j, '@source_node': pre, '@target_node': post})
                    possible_connections = possible_connections.drop(j)
                    break
    
        to_connect = pd.DataFrame(matched).set_index('index')
        connections[['@source_node', '@target_node']] =\
            to_connect[['@source_node', '@target_node']]
    assert(len(connections) == num_conns)
    return connections


# TODO: add tests
def _shift_synapses(new_edges, based_on):
    new_edges = new_edges.set_index(['@source_node', '@target_node']).sort_index()
    new_edges['r'] = based_on
    new_edges = new_edges.reset_index()
    sorted_by_conductance = new_edges.sort_values(['@target_node', 'conductance'], ascending=False)
    sorted_by_r = new_edges.sort_values(['@target_node', 'r'], ascending=False)
    synaptic_parameters = [c for c in sorted_by_r if c not in ['r', '@source_node', '@target_node']]
    sorted_by_r[synaptic_parameters] = sorted_by_conductance[synaptic_parameters].values

    return sorted_by_r.sort_values(['@target_node', '@source_node'])


def _retrieve_synapse_properties(prev_edges, prev_pairs, new_pairs):
    indexed_by_prev_pair = prev_edges.set_index(['@source_node', '@target_node']).sort_index()
    new_by_prev = new_pairs.set_index(list(prev_pairs.values.transpose()))
    indexed_by_prev_pair[['@source_node', '@target_node']] = new_by_prev
    return indexed_by_prev_pair.reset_index(drop=True)


def _place_synapses_from_structural(new_edges, structural_edges, new_connections):
    new_connections = new_connections.set_index(['@source_node', '@target_node'])
    struct_by_nsyn = structural_edges.set_index(['@source_node', '@target_node']).groupby(new_connections['nsyn'])
    edges_by_nsyn = new_edges.set_index(['@source_node', '@target_node']).sort_index()
    out = []
    for nsyn, edges in edges_by_nsyn.groupby(new_connections['nsyn']):
        struct = struct_by_nsyn.get_group(nsyn).reset_index()
        struct_edges = struct.groupby(['@source_node', '@target_node']).sample(n=int(nsyn)).set_index(['@source_node', '@target_node']).sort_index()
        edges = edges.sort_index()
        edges[struct_edges.columns] = struct_edges.values
        out.append(edges)
    return pd.concat(out).reset_index()
            

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
