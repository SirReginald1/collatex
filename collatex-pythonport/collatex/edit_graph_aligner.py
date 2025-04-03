'''
Created on Aug 5, 2014

@author: Ronald Haentjens Dekker
'''
from enum import Enum
from typing import List, Dict, Any, Callable, Union, TYPE_CHECKING
from collatex.core_classes import CollationAlgorithm, VariantGraphRanking, VariantGraph, Collation, Witness, Token, VariantGraphVertex
from collatex.tokenindex import TokenIndex
from collatex.transposition_handling import TranspositionDetection
if TYPE_CHECKING:
    from collatex.suffix_based_scorer import Scorer as SufixScorer
    from collatex.transposition_handling import TranspositionDetection
    from collatex.experimental_astar_aligner import AstarEditGraphNode
    from collatex.block import Block, Instance


class EditGraphNode(object):
    def __init__(self):
        self.g = 0  # global score
        self.segments = 0  # number of segments
        self.match = False  # this node represents a match or not

    def __repr__(self):
        return repr(self.g)

    '''
    Aligner based on an edit graph.
    It needs a g function and a definition of a match.
    Since every node of the graph has three children the graph is represented as a table internally.
    Default implementation is a* based.
    '''


class Match(object):
    """Stores the vertex in a graph and the token in a witness of a match."""
    def __init__(self, vertex, token):
        self.vertex: VariantGraphVertex = vertex
        """The vertex that represents the match."""
        self.token: Token = token
        """The token that represents a match."""

    def __repr__(self):
        return str.format("Match(vertex={},token={}", self.vertex, self.token)


class MatchCoordinate():
    """Stores the coordinates of the a match in both the witness sequence an the graph rank."""
    def __init__(self, row, rank):
        self.index: int = row  # position in witness, starting from zero
        """Position of match in sequences sequence. (start of match?????)"""
        self.rank = rank  # rank in the variant graph
        """Position of match in the graph??????????"""

    def __eq__(self, other):
        """A match is equal if both there position in the sequence is the same and 
        there rank in the graph.

        ### Param:
            - other (MatchCoordinate): The other MatchCoordinate to be compared to.
        """
        return self.index == other.index and self.rank == other.rank

    def __hash__(self):
        """The hash is given by: 10 * self.index + self.rank"""
        return 10 * self.index + self.rank

    def __repr__(self):
        return str.format("MatchCoordinate(index = {}, rank = {})", self.index, self.rank)


class MatchCube():
    """Object that is used to determine token block matches and store them."""
    def __init__(self, 
                 token_index: TokenIndex, 
                 witness: Witness, 
                 vertex_array: VariantGraphVertex, 
                 variant_graph_ranking: VariantGraphRanking, 
                 properties_filter: Callable) -> "MatchCube":
        """
        
        ### Args:
            - properties_filter (Callable, Optional): A function that takes 2 token_data objects and returns a boolean that indicates 
            if the tokens are a match. The default value of the match is true so the purpuse of this function is to disprouve matches based
            on extra information.
        """
        self.matches: Dict[MatchCoordinate, Match] = {}
        """Links all existing matches with there coordinates."""
        # print("> vertex_array =", vertex_array)
        start_token_position_for_witness = token_index.start_token_position_for_witness(witness)
        # print("> start_token_position_for_witness=", start_token_position_for_witness)
        # Get list of all block instances associated with given witness
        instances = token_index.block_instances_for_witness(witness)
        # print("> token_index.witness_to_block_instances", token_index.witness_to_block_instances)
        # print("> instances", instances)
        for witness_instance in instances:
            # print("> witness_instance=", witness_instance)
            block: Block = witness_instance.block
            all_instances: List[Instance] = block.get_all_instances() # Get all instance of this block of tokens
            """List of all instances of the selcted block"""
            # Look through all instances of the selected block and select only those that start befor the one from the 
            # curently selected witness
            graph_instances = [i for i in all_instances if i.start_token < start_token_position_for_witness]
            # In the graph run through all selected intances 
            for graph_instance in graph_instances:
                graph_start_token = graph_instance.start_token
                # For each token in the block instance
                for i in range(0, block.length):
                    # print("> graph_start_token + i =", (graph_start_token + i))
                    v = vertex_array[graph_start_token + i]
                    if v is None:
                        raise Exception(
                            str.format('Vertex is null for token {} {} that is supposed to be mapped to a vertex in'
                                       ' the graph!', graph_start_token, i))

                    rank = variant_graph_ranking.apply(v) - 1 # Get the rank - 1 of the vertex
                    witness_start_token = witness_instance.start_token + i
                    row = witness_start_token - start_token_position_for_witness
                    token = token_index.token_array[witness_start_token]
                    match = True
                    if properties_filter:
                        other = token_index.token_array[graph_start_token]
                        token_data1 = self.filtered_token_data(token)
                        token_data2 = self.filtered_token_data(other)
                        match = properties_filter(token_data1, token_data2)
                    if match:
                        match = Match(v, token)
                        coordinate = MatchCoordinate(row, rank)
                        self.matches[coordinate] = match

    @staticmethod
    def filtered_token_data(token: Token) -> Dict[str, Any]:
        """Prepaires token data for the token filter function by removing the sigil and token array position from the token data dictionary."""
        token_data1 = dict(token.token_data)
        del token_data1['_sigil']
        del token_data1['_token_array_position']
        return token_data1

    @staticmethod
    def has_tokens(vertex):
        return not vertex.tokens().isEmpty()

    def has_match(self, y, x):
        c = MatchCoordinate(y, x)
        return c in self.matches

    def get_match(self, y, x):
        c = MatchCoordinate(y, x)
        return self.matches[c]


class ScoreType(Enum):
    match = 1
    mismatch = 2
    addition = 3
    deletion = 4
    empty = 5


class Score():
    def __init__(self, 
                 score_type: int, 
                 x: int, 
                 y: int, 
                 parent: "Score", 
                 global_score=None) -> "Score":
        self.type: int = score_type
        self.x: int = x
        self.y: int = y
        self.parent: "Score" = parent
        self.previous_x: int = 0 if (parent is None) else parent.x
        self.previous_y: int = 0 if (parent is None) else parent.y
        self.global_score = parent.global_score if global_score is None else global_score

    def __repr__(self):
        return str.format("({},{})", self.global_score, self.type.name)


class Scorer():
    """Class used to build the scores from the match cube 
    based on edits made to graph in order to make witnesses match. """
    def __init__(self, match_cube: Union[MatchCube, None]=None) -> "Scorer":
        self.match_cube: Union[MatchCube, None] = match_cube
        """The reference of matches used by the scorer. ????????"""

    def gap(self, x: int, y: int, parent: Score) -> Score:
        score_type = self.determine_type(x, y, parent)
        return Score(score_type, x, y, parent, parent.global_score - 1)

    def score(self, x: int, y: int, parent: Score) -> Score:
        rank = x - 1
        if self.match_cube.has_match(y - 1, rank):
            return Score(ScoreType.match, x, y, parent, parent.global_score + 1)
        return Score(ScoreType.mismatch, x, y, parent, parent.global_score - 1)

    @staticmethod
    def determine_type(x: int, y: int, parent: Score) -> int:
        """Determins the type of gap in the sequence. If it is an addition or a deletion."""
        if x == parent.x:
            return ScoreType.addition
        if y == parent.y:
            return ScoreType.deletion
        return ScoreType.empty


class ScoreIterator():
    def __init__(self, score_matrix):
        self.score_matrix = score_matrix
        self.x = len(score_matrix[0]) - 1
        self.y = len(score_matrix) - 1

    def __iter__(self):
        return self

    def _has_next(self):
        return not (self.x == 0 and self.y == 0)

    def __next__(self):
        if self._has_next():
            current_score = self.score_matrix[self.y][self.x]
            self.x = current_score.previous_x
            self.y = current_score.previous_y
            return current_score
        else:
            raise StopIteration()


class EditGraphAligner(CollationAlgorithm):
    """The aligner presented in Dekker's paper. """
    def __init__(self, 
                 collation: Collation, 
                 near_match: bool = False, 
                 debug_scores: bool = False, 
                 detect_transpositions: bool = False,
                 properties_filter: Callable = None):
        self.scorer = Scorer()
        self.collation = collation
        self.debug_scores = debug_scores
        self.detect_transpositions = detect_transpositions
        self.properties_filter = properties_filter
        self.token_index: TokenIndex = TokenIndex(collation.witnesses)
        """Object used to store lists of prefixes, suffixes, and token indexes to be used in collation."""
        self.token_position_to_vertex: Dict[int, VariantGraphVertex] = {}
        """Links token index in sequence to its VariantGraphVertex in the associated VariantGraph."""
        self.added_witness = []
        self.omitted_base = []
        self.vertex_array: List[VariantGraphVertex] = []
        """List of all vertexes involved in the allignement. ??????????????"""
        self.cells = [[]]

    def collate(self, graph: VariantGraph) -> None:
        """
        :type graph: VariantGraph
        """
        # prepare the token index
        self.token_index.prepare()
        self.vertex_array: List[int] = [None] * len(self.token_index.token_array)
        """Array that links the token positions in the sequence to vertex in VariantGraph."""

        # Build the variant graph for the first witness
        # this is easy: generate a vertex for every token
        first_witness: Witness = self.collation.witnesses[0]
        tokens = first_witness.tokens()
        token_to_vertex = self.merge(graph, first_witness.sigil, tokens)
        # print("> token_to_vertex=", token_to_vertex)
        self.update_token_position_to_vertex(token_to_vertex)
        self.update_token_to_vertex_array(tokens, first_witness, self.token_position_to_vertex)

        # align witness 2 - n
        for x in range(1, len(self.collation.witnesses)):
            witness: Witness = self.collation.witnesses[x]
            tokens = witness.tokens()
            # print("\nwitness", witness.sigil)

            variant_graph_ranking = VariantGraphRanking.of(graph)
            # print("> x =", x, ", variant_graph_ranking =", variant_graph_ranking.byRank)
            variant_graph_ranks = list(set(map(lambda v: variant_graph_ranking.byVertex.get(v), graph.vertices())))
            # we leave in the rank of the start vertex, but remove the rank of the end vertex
            variant_graph_ranks.pop()

            # now the vertical stuff
            tokens_as_index_list = self.as_index_list(tokens)

            match_cube = MatchCube(self.token_index, 
                                   witness, 
                                   self.vertex_array, 
                                   variant_graph_ranking,
                                   self.properties_filter)
            # print("> match_cube.matches=", match_cube.matches)
            self.fill_needleman_wunsch_table(variant_graph_ranks, tokens_as_index_list, match_cube)

            aligned = self.align_matching_tokens(match_cube)
            print(f"edit_graph_aligner.collate: aligned dict: {aligned}")
            # print("> aligned=", aligned)
            # print("self.token_index.token_array=", self.token_index.token_array)
            # alignment = self.align_function(superbase, next_witness, token_to_vertex, match_cube)

            # merge
            witness_token_to_generated_vertex = self.merge(graph, witness.sigil, witness.tokens(), aligned)
            # print("> witness_token_to_generated_vertex =", witness_token_to_generated_vertex)
            token_to_vertex.update(witness_token_to_generated_vertex)
            # print("> token_to_vertex =", token_to_vertex)
            self.update_token_position_to_vertex(token_to_vertex, aligned)
            witness_token_position_to_vertex = {}
            for p in self.token_index.get_range_for_witness(witness.sigil):
                # print("> p= ", p)
                witness_token_position_to_vertex[p] = self.token_position_to_vertex[p]
            self.update_token_to_vertex_array(tokens, witness, witness_token_position_to_vertex)
            # print("> vertex_array =", self.vertex_array)

            #             print("actual")
            #             self._debug_edit_graph_table(self.table)
            #             print("expected")
            #             self._debug_edit_graph_table(self.table2)

            # change superbase
            # superbase = self.new_superbase

            if self.detect_transpositions:
                detector = TranspositionDetection(self)
                detector.detect()

                # if self.debug_scores:
                #     self._debug_edit_graph_table(self.table)

    @staticmethod
    def as_index_list(tokens) -> List[int]:
        """Returns a list of integers from 0 to nb tokens."""
        tokens_as_index_list = [0]
        counter = 1
        for t in tokens:
            tokens_as_index_list.append(counter)
            counter += 1
        return tokens_as_index_list

    def fill_needleman_wunsch_table(self, 
                                    variant_graph_ranks: VariantGraphRanking, 
                                    tokens_as_index_list, 
                                    match_cube: MatchCube):
        self.cells = [[None for row in range(0, len(variant_graph_ranks))] for col in
                      range(0, len(tokens_as_index_list))]
        scorer = Scorer(match_cube)

        # init 0,0
        self.cells[0][0] = Score(ScoreType.empty, 0, 0, None, 0)

        # fill the first row with gaps
        for x in range(1, len(variant_graph_ranks)):
            previous_x = x - 1
            self.cells[0][x] = scorer.gap(x, 0, self.cells[0][previous_x])

        # fill the first column with gaps
        for y in range(1, len(tokens_as_index_list)):
            # print("\nself.cells.len = ", len(self.cells), " x ", len(self.cells[0]))
            # print("y=", y)
            # print("self.cells[y][0]=", self.cells[y][0])
            previous_y = y - 1
            # print("previous_y=", previous_y)
            # print("self.cells[previous_y][0]=", self.cells[previous_y][0])
            self.cells[y][0] = scorer.gap(0, y, self.cells[previous_y][0])

        _debug_cells(self.cells)

        # fill the remaining cells
        # fill the rest of the cells in a y by x fashion
        for y in range(1, len(tokens_as_index_list)):
            for x in range(1, len(variant_graph_ranks)):
                previous_y = y - 1
                previous_x = x - 1
                from_upper_left = scorer.score(x, y, self.cells[previous_y][previous_x])
                from_left = scorer.gap(x, y, self.cells[y][previous_x])
                from_upper = self.calculate_from_upper(scorer, y, x, previous_y, match_cube)
                max_score = max(from_upper_left, from_left, from_upper, key=lambda s: s.global_score)
                self.cells[y][x] = max_score

    def calculate_from_upper(self, 
                             scorer: Scorer, 
                             y, 
                             x, 
                             previous_y, 
                             match_cube: MatchCube):
        upper_is_match = match_cube.has_match(previous_y - 1, x - 1)
        if upper_is_match:
            return scorer.score(x, y, self.cells[previous_y][x])
        else:
            return scorer.gap(x, y, self.cells[previous_y][x])

    def align_matching_tokens(self, cube: MatchCube) -> Dict[Token, VariantGraphVertex]:
        #  using the score iterator..
        #  find all the matches
        #  later for the transposition detection, we also want to keep track of all the additions,
        #  omissions, and replacements
        aligned: Dict[Token, VariantGraphVertex] = {}
        scores = ScoreIterator(self.cells)
        matched_vertices = []
        for score in scores:
            if score.type == ScoreType.match:
                rank = score.x - 1
                match = cube.get_match(score.y - 1, rank)
                if match.vertex not in matched_vertices:
                    aligned[match.token] = match.vertex
                    matched_vertices.append(match.vertex)
        return aligned

    def update_token_to_vertex_array(self, tokens, witness, witness_token_position_to_vertex):
        # we need to update the token -> vertex map
        # that information is stored in protected map
        # print("> witness_token_position_to_vertex =", witness_token_position_to_vertex)
        # t = list(witness_token_to_vertex)[0]
        # #print("> list(witness_token_to_vertex)[0] =", t)
        # #print("> t.token_string =", t.token_string)
        # #print("> t.token_data =", t.token_data)
        # print("> witness_token_position_to_vertex =", witness_token_position_to_vertex)
        for token_position in self.token_index.get_range_for_witness(witness.sigil):
            # print("> token_position =", token_position)
            vertex = witness_token_position_to_vertex[token_position]
            self.vertex_array[token_position] = vertex

    def update_token_position_to_vertex(self, token_to_vertex: Dict[Token, VariantGraphVertex], aligned: Dict[Token, Any]={}) -> None:
        """Links token position index to vertex then overwirte with aligned tokens"""
        for token in token_to_vertex:
            # print("> token =", token)
            position = token.token_data['_token_array_position']
            # print("> position =", position)
            self.token_position_to_vertex[position] = token_to_vertex[token]
        for token in aligned:
            # print("> token =", token)
            position = token.token_data['_token_array_position']
            # print("> position =", position)
            self.token_position_to_vertex[position] = aligned[token]

            # print("> self.token_position_to_vertex=", self.token_position_to_vertex)


def _debug_cells(cells):
    y = 0
    # for row in cells:
    #     x = 0
    #     print()
    #     for cell in row:
    #         if cell is not None:
    #             print(str.format("[{},{}]:{}", x, y, cell))
    #         x += 1
    #     y += 1
