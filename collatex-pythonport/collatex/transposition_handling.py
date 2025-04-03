'''
    Created on May 3, 2014
        
    @author: Ronald Haentjens Dekker
'''
from typing import Set, Union, List, Dict, Tuple, Optional, TYPE_CHECKING
from collatex.core_classes import VariantGraphRanking

#from collections import defaultdict
from math import fabs
from collatex.core_classes import Token
from functools import cmp_to_key

if TYPE_CHECKING:
    from collatex.edit_graph_aligner import Match
    from collatex.edit_graph_aligner import EditGraphAligner
    from collatex.experimental_astar_aligner import AstarEditGraphAligner
    from collatex.block import Instance

# New transposition detection implementation, still beta quality
# This implementation works with the new collation algorithm (LCP intervals and edit graph)
class TranspositionDetection(object):
    """Object used for transposition detection from CollationAlgorithm objects.\n 
    !!!!!!!!!!!!!! ONLY WORKS WITH EditGraphAligner and maybe AstarEditGraphAligner !!!!!!!!!!!!!
    """

    def __init__(self, aligner: Union["EditGraphAligner", "AstarEditGraphAligner"]) -> "TranspositionDetection":
        self.aligner: Union["EditGraphAligner", "AstarEditGraphAligner"] = aligner
        """The aligner being used. !!!! We know it's this class as transposition detection 
        only occurse in Dekkers edit Graph algorithm !!!!!!!"""


    def detect_experimental(self, phrase_matches_witness_order: Optional[List[List["Match"]]], base: 'VariantGraph') -> List[List["Match"]]:
        # If there are no phrase matches it is not possible
        # to detect transpositions, return an empty list
        if not phrase_matches_witness_order:
            return []

        print(f"detect_experimental: WO (input): {phrase_matches_witness_order}")
        # Order phrase matches by topological order (rank), then by witness order
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #print(f"detect_experimental: param1: {phrase_matches_witness_order}, param2: {base}")
        #print(f"detect_experimental: param1 type: {type(phrase_matches_witness_order)}, param2 type: {type(base)}")
        #print(f"detect_experimental: param1 depth1 type: {type(phrase_matches_witness_order[0])}")
        #print(f"detect_experimental: param1  depth2 type: {type(phrase_matches_witness_order[0][0])}")
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ranking = self.rank_the_graph(phrase_matches = phrase_matches_witness_order, base = base)
        print(f"detect_experimental: ranking.byVertex: {ranking.byVertex}")
        print(f"detect_experimental: ranking.byRank: {ranking.byRank}")
        #print("detect_experimental: ranking: ", end="")
        #for e in ranking:
        #    print( f"{e}", end=", ")
        #print("")

        def comp(pm1: List["Match"], pm2: List["Match"]) -> int:
            rank1 = ranking.apply(pm1[0].vertex)
            rank2 = ranking.apply(pm2[0].vertex)
            difference = rank1 - rank2
            if difference != 0:
                return difference
            index1 = phrase_matches_witness_order.index(pm1)
            index2 = phrase_matches_witness_order.index(pm2)
            return index1 - index2

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #print(f"detect_experimental: phrase_matches_witness_order (before sort): {phrase_matches_witness_order}")
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # List of match lists in order of apparition in the graph
        phrase_matches_graph_order = sorted(phrase_matches_witness_order, key = cmp_to_key(comp))
        print(f"detect_experimental: GO: {phrase_matches_graph_order}")

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #print(f"detect_experimental: phrase_matches_graph_order (after sort): {phrase_matches_graph_order}")
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Map 1
        #phrase_matches_graph_order_idx_ref = range(len(phrase_matches_graph_order))
        # NOTE: Convert list to tuple as tuples are imutable and can be hashed
        phrase_match_to_graph_index: Dict[Tuple["Match"], int] = {
            tuple(match): i for i, match in enumerate(phrase_matches_graph_order)
        }
        print(f"detect_experimental: TGI: {phrase_match_to_graph_index}")
        #print("detect_experimental: TGI: ", end="")
        #for key in phrase_match_to_graph_index:
        #    print( f"{phrase_match_to_graph_index[key]}", end=", ")
        #print("")
        
        # NOTE: This is at worst of speed (n(n-1)/2). Implement custom sorting algorithm or use custom comparison
        # NOTE: somhow to keep track of indexes during sorting to speed up process
        #phrase_match_to_graph_index = {}
        #for idx in range(len(phrase_matches_witness_order)):
        #    phrase_match_to_graph_index[idx] = phrase_matches_graph_order.index(phrase_matches_witness_order[idx])

        # Index lists
        phrase_matches_witness_index: List[int] = list(range(len(phrase_matches_witness_order)))
        print(f"detect_experimental: WI: {phrase_matches_witness_index}")

        phrase_matches_graph_index: List[Optional[int]] = [
            phrase_match_to_graph_index[tuple(pm)] for pm in phrase_matches_witness_order
        ]
        print(f"detect_experimental: GI: {phrase_matches_graph_index}")
        

        # Result containers
        non_transposed_phrase_matches: List[List["Match"]] = list(phrase_matches_witness_order)
        print(f"detect_experimental: NTPM: {non_transposed_phrase_matches}")
        transpositions: List[List["Match"]] = []

        print("############## Main loop resolving transposition ########################")

        # Main loop: resolve transpositions
        while True:
            # Map 2
            phrase_match_to_distance_map: Dict[Tuple["Match"], int] = {}
            print("########### Start for loop: ##############")
            for i in range(len(non_transposed_phrase_matches)):
                print(f"i: {i}")
                graph_index = phrase_matches_graph_index[i]
                print(f"graph_index: {graph_index}")
                witness_index = phrase_matches_witness_index[i]
                print(f"witness_index: {witness_index}")
                distance = abs(graph_index - witness_index)
                print(f"distance: {distance}")
                phrase_match = non_transposed_phrase_matches[i]
                print(f"phrase_match: {phrase_match}")
                phrase_match_to_distance_map[tuple(phrase_match)] = distance
                print(f"phrase_match_to_distance_map[tuple(phrase_match)]: {phrase_match_to_distance_map[tuple(phrase_match)]}")
            print("################ End loop ################")

            distance_list = list(phrase_match_to_distance_map.values())
            print(f"distance_list: {distance_list}")
            if not distance_list or max(distance_list) == 0:
                print("!!!!!!! Enter end function condition. !!!!!!!!")
                break

            # Sort by distance desc, then size
            def comp2(pm1: List["Match"], pm2: List["Match"]) -> int:
                distance1 = phrase_match_to_distance_map[tuple(pm1)]
                distance2 = phrase_match_to_distance_map[tuple(pm2)]
                difference = distance2 - distance1
                if difference != 0:
                    return difference
                return self.determine_size(pm1) - self.determine_size(pm2)

            sorted_phrase_matches = sorted(non_transposed_phrase_matches, key=cmp_to_key(comp2))
            print(f"sorted_phrase_matches: {sorted_phrase_matches}")
            transposed_phrase = sorted_phrase_matches.pop(0)
            print(f"transposed_phrase: {transposed_phrase}")
            transposed_index = phrase_match_to_graph_index[tuple(transposed_phrase)]
            print(f"transposed_index: {transposed_index}")
            graph_index = phrase_matches_graph_index.index(transposed_index)
            print(f"graph_index: {graph_index}")
            transposed_with_index = phrase_matches_witness_index[graph_index]
            print(f"transposed_with_index: {transposed_with_index}")
            linked_transposed_phrase = phrase_matches_graph_order[transposed_with_index]
            print(f"linked_transposed_phrase: {linked_transposed_phrase}")

            self.add_transposition(
                phrase_match_to_graph_index,
                phrase_matches_witness_index,
                phrase_matches_graph_index,
                non_transposed_phrase_matches,
                transpositions,
                transposed_phrase
            )
            print("After add_transposition:")
            print(f"ltp: {phrase_match_to_graph_index}")
            print(f"WI: {phrase_match_to_graph_index}")
            print(f"GI: {phrase_match_to_graph_index}")
            print(f"NTPM: {phrase_match_to_graph_index}")
            print(f"transposition: {phrase_match_to_graph_index}")
            print(f"TP: {phrase_match_to_graph_index}")

            distance = phrase_match_to_distance_map[tuple(transposed_phrase)]
            print(f"distance: {distance}")
            if (distance == phrase_match_to_distance_map[tuple(linked_transposed_phrase)]
                    and distance > 1):
                print("In if statment: ")
                self.add_transposition(
                    phrase_match_to_graph_index,
                    phrase_matches_witness_index,
                    phrase_matches_graph_index,
                    non_transposed_phrase_matches,
                    transpositions,
                    linked_transposed_phrase
                )
                print("After add_transposition:")
                print(f"ltp: {phrase_match_to_graph_index}")
                print(f"WI: {phrase_match_to_graph_index}")
                print(f"GI: {phrase_match_to_graph_index}")
                print(f"NTPM: {phrase_match_to_graph_index}")
                print(f"transposition: {phrase_match_to_graph_index}")
                print(f"TP: {phrase_match_to_graph_index}")

        return transpositions

    def add_transposition(self,
                          phrase_match_to_index: Dict[List["Match"], int],
                          phrase_witness_ranks: List[Optional[int]],
                          phrase_graph_ranks: List[Optional[int]],
                          non_transposed_phrase_matches: List[List["Match"]],
                          transpositions: List[List["Match"]],
                          transposed_phrase: List["Match"]) -> None:
        index_to_remove = phrase_match_to_index[transposed_phrase]
        non_transposed_phrase_matches.remove(transposed_phrase)
        transpositions.append(transposed_phrase)
        phrase_graph_ranks.pop(index_to_remove)
        phrase_witness_ranks.pop(index_to_remove)


    def rank_the_graph(self, 
                       phrase_matches: List[List["Match"]],
                       base: 'VariantGraph') -> 'VariantGraphRanking':
        # Collect all the starting vertices from each phrase
        matched_vertices = set()
        for phrase_match in phrase_matches:
            matched_vertices.add(phrase_match[0].vertex)

        return VariantGraphRanking.of_only_certain_vertices(base, matched_vertices)


    def determine_size(self, t: List["Match"]) -> int:
        first_match = t[0]
        if not isinstance(first_match.token, Token):
            return len(t)

        char_length = 0
        for m in t:
            token = m.token  # Assumes token is a SimpleToken
            char_length += len(token.normalized)

        return char_length


    def detect(self):
        # analyse additions and omissions to detect transpositions
        # We fetch all the occurrences of the added tokens
        # Using the scorer (which has the blocks and occurrences of these blocks)
        added_occurrences: Set[Instance] = set()
        # Run through all tokens that are classified as additions
        for token in self.aligner.additions:
            # get the occurrence / occurrence list (of block ????) associated with that token
            occurrence = self.aligner.scorer.global_tokens_to_occurrences[token] # Find the specific scorers way of linking tokens to Instances
            # NOTE: not every token is an occurrence of a block
            if occurrence:
                added_occurrences.add(occurrence)
        # for every occurrences we have to detect the associated block
        added_blocks = set()
        added_blocks_dict = {}
        for occurrence in added_occurrences:
            added_blocks.add(occurrence.block)
            added_blocks_dict[occurrence.block]=occurrence
        print("Added blocks: "+str(added_blocks))
        # Fetch all omitted block
        omitted_occurrences = set()
        for token in self.aligner.omissions:
            # get occurrences from scorer
            occurrence = self.aligner.scorer.global_tokens_to_occurrences[token]
            if occurrence:
                omitted_occurrences.add(occurrence)
        # for every occurrences we have to detect the associated block
        omitted_blocks = set()
        omitted_blocks_dict = {}
        for occurrence in omitted_occurrences:
            omitted_blocks.add(occurrence.block)
            omitted_blocks_dict[occurrence.block]=occurrence
        print("omitted blocks: "+str(omitted_blocks))

        # calculate transpositions by taking the intersection of the two sets
        transposed_blocks = omitted_blocks.intersection(added_blocks)
        print ("transposed blocks: "+str(transposed_blocks))


        # for now assume that there is only one occurrence for every block
        # otherwise we skip
        for block in transposed_blocks:
            occurrence1 = added_blocks_dict[block]
            occurrence2 = omitted_blocks_dict[block]
            # we need to go from the occurrences to the tokens
            token_positions = zip(occurrence1.token_range, occurrence2.token_range)
            for (token_position_base, token_position_witness) in token_positions:
                token_base = self.aligner.collation.tokens[token_position_base]
                token_witness = self.aligner.collation.tokens[token_position_witness]
                print(token_base, token_witness)








#===========================================================================
# Direct port from Java code
#===========================================================================
class PhraseMatchDetector(object):
    def _add_new_phrase_match_and_clear_buffer(self, phrase_matches, base_phrase, witness_phrase):
        if base_phrase:
            phrase_matches.append(zip(base_phrase, witness_phrase)) 
            del base_phrase[:]
            del witness_phrase[:]

    def detect(self, linked_tokens, base, tokens):
        phrase_matches = []
        base_phrase = []
        witness_phrase = []
        previous = base.start

        for token in tokens:
            if not token in linked_tokens:
                self._add_new_phrase_match_and_clear_buffer(phrase_matches, base_phrase, witness_phrase)
                continue
            base_vertex = linked_tokens[token]
            # requirements:
            # - see comments in java class
            same_transpositions = True #TODO
            same_witnesses = True #TODO
            directed_edge = base.edge_between(previous, base_vertex)
            is_near = same_transpositions and same_witnesses and directed_edge and len(base.out_edges(previous))==1 and len(base.in_edges(base_vertex))==1
            if not is_near:
                self._add_new_phrase_match_and_clear_buffer(phrase_matches, base_phrase, witness_phrase)
            base_phrase.append(base_vertex)
            witness_phrase.append(token)
            previous = base_vertex
        if base_phrase:
            phrase_matches.append(zip(base_phrase, witness_phrase)) 
        return phrase_matches

#=================================================
# Almost fully direct port from Java code
#=================================================
class TranspositionDetector(object):
    def detect(self, phrasematches, base):
        if not phrasematches:
            return []

        ranking = self._rank_the_graph(phrasematches, base)

        def compare_phrasematches(pm1, pm2):
            (vertex1, _) = pm1[0]
            (vertex2, _) = pm2[0]
            rank1 = ranking.apply(vertex1)
            rank2 = ranking.apply(vertex2)
            difference = rank1 - rank2

            if difference != 0:
                return difference
            index1 = phrasematches.index(pm1)
            index2 = phrasematches.index(pm2)
            return index1 - index2

        phrasematches_graph_order = sorted(phrasematches, cmp=compare_phrasematches)

        # map 1
        self.phrasematch_to_index = {}
        for idx, val in enumerate(phrasematches_graph_order):
            self.phrasematch_to_index[val[0]]=idx

        # We calculate the index for all the phrase matches
        # First in witness order, then in graph order
        phrasematches_graph_index = range(0, len(phrasematches))

        phrasematches_witness_index = []
        for phrasematch in phrasematches:
            phrasematches_witness_index.append(self.phrasematch_to_index[phrasematch[0]])

        # initialize result variables
        non_transposed_phrasematches = list(phrasematches)
        transpositions = []

        # loop here until the maximum distance == 0
        while(True):
            # map 2
            phrasematch_to_distance = {}
            for i, phrasematch in enumerate(non_transposed_phrasematches):
                graph_index = phrasematches_graph_index[i]
                witness_index = phrasematches_witness_index[i]
                distance = abs(graph_index - witness_index)
                phrasematch_to_distance[phrasematch[0]]=distance

            distance_list = list(phrasematch_to_distance.values())

            if not distance_list or max(distance_list) == 0:
                break

            def comp2(pm1, pm2):
                # first order by distance
                distance1 = phrasematch_to_distance[pm1[0]]
                distance2 = phrasematch_to_distance[pm2[0]]
                difference = distance2 - distance1
                if difference != 0:
                    return difference

                # second order by size
                #TODO: this does not work for Greek texts with lots of small words!
                #TODO: it should determine which block this phrasematch is part of and
                #TODO: the number of occurrences for that block
                return len(pm1) - len(pm2)

            sorted_phrasematches = sorted(non_transposed_phrasematches, cmp = comp2) 
            transposedphrase = sorted_phrasematches[0]

            transposed_index = self.phrasematch_to_index[transposedphrase[0]]
            graph_index = phrasematches_graph_index.index(transposed_index)
            transposed_with_index = phrasematches_witness_index[graph_index]
            linked_transposed_phrase = phrasematches_graph_order[transposed_with_index]

            self._add_transposition(phrasematches_witness_index, phrasematches_graph_index, non_transposed_phrasematches, transpositions, transposedphrase)

            distance = phrasematch_to_distance[transposedphrase[0]]
            if distance == phrasematch_to_distance[linked_transposed_phrase[0]] and distance > 1:
                self._add_transposition(phrasematches_witness_index, phrasematches_graph_index, non_transposed_phrasematches, transpositions, linked_transposed_phrase)

        return transpositions

    def _add_transposition(self, phrasematches_witness_index, phrasematches_graph_index, non_transposed_phrasematches, transpositions, transposed_phrase):
        index_to_remove = self.phrasematch_to_index[transposed_phrase[0]]
        non_transposed_phrasematches.remove(transposed_phrase)
        transpositions.append(transposed_phrase)
        phrasematches_graph_index.remove(index_to_remove)
        phrasematches_witness_index.remove(index_to_remove)

    def _rank_the_graph(self, phrase_matches, base):
        #TODO: rank the graph based on only the first vertex of each of the phrasematches!
        return VariantGraphRanking.of(base)