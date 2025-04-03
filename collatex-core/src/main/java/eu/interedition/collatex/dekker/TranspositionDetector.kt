/*
 * Copyright (c) 2015 The Interedition Development Group.
 *
 * This file is part of CollateX.
 *
 * CollateX is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CollateX is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CollateX.  If not, see <http://www.gnu.org/licenses/>.
 */
package eu.interedition.collatex.dekker

import eu.interedition.collatex.VariantGraph
import eu.interedition.collatex.simple.SimpleToken
import eu.interedition.collatex.util.VariantGraphRanking
import java.util.*
import kotlin.Comparator
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.HashSet
import kotlin.collections.LinkedHashMap
import kotlin.math.abs

/**
 * @author Ronald Haentjens Dekker
 */
class TranspositionDetector {

    /**
     * Detect transpositions in already aligned sequences.
     * 
     * @param phraseMatchesWitnessOrder A list of matche list for each witness in the collation.
     * @param base The VariantGraph of the aligned witness sequences.
     */
    fun detect(phraseMatchesWitnessOrder: List<List<Match>>?, base: VariantGraph): MutableList<List<Match>> {
        // if there are no phrase matches it is not possible
        // to detect transpositions, return an empty list
        if (phraseMatchesWitnessOrder!!.isEmpty()) {
            return ArrayList()
        }

        /*
         * We order the phrase matches in the topological order
         * of the graph (called rank). When the rank is equal
         * for two phrase matches, the witness order is used
         * to differentiate.
         */
        val ranking = rankTheGraph(phraseMatchesWitnessOrder, base)

        /*
         * Get the difference between the rank of the vertex from the first match in list 1 and 2.
         * If there is no difference between the ranks find the index of the first occurence of both lists in phraseMatchesWitnessOrder
         * and return at the difference between there indexes.
         */
        val comp = Comparator { pm1:List<Match>, pm2: List<Match> ->
            val rank1 = ranking.apply(pm1[0].vertex)
            val rank2 = ranking.apply(pm2[0].vertex)
            val difference = rank1 - rank2
            when {
                difference != 0 -> difference
                else -> {
                    val index1 = phraseMatchesWitnessOrder.indexOf(pm1)
                    val index2 = phraseMatchesWitnessOrder.indexOf(pm2)
                    index1 - index2
                }
            }
        }

        // Sort phraseMatchesWitnessOrder using comp function. Is used to map lists of matches to indexes in there coresponding graph
        val phraseMatchesGraphOrder: List<List<Match>> = phraseMatchesWitnessOrder.sortedWith(comp)

        // Map 1
        val phraseMatchToGraphIndex: MutableMap<List<Match>, Int> = HashMap()
        for (i in phraseMatchesGraphOrder.indices) {
            phraseMatchToGraphIndex[phraseMatchesGraphOrder[i]] = i
        }

        /*
         * We calculate the index for all the phrase matches
         * All the indexes are stored in the order of the witnesses arrays as this allows distance comparison 
         * by iterating along the witness list indexes
         */
        val phraseMatchesWitnessIndex: MutableList<Int?> = ArrayList()
        val phraseMatchesGraphIndex: MutableList<Int?> = ArrayList()
        for (i in phraseMatchesWitnessOrder.indices) {
            phraseMatchesWitnessIndex.add(i)
        }
        for (phraseMatch in phraseMatchesWitnessOrder) {
            phraseMatchesGraphIndex.add(phraseMatchToGraphIndex[phraseMatch])
        }

        // DEBUG
        // println(phraseMatchesGraphIndex)
        // println(phraseMatchesWitnessIndex)

        /*
         * Initialize result variables
         */
        // This array is a copy of the input array and will have it's values progressively removed from it.
        val nonTransposedPhraseMatches: MutableList<List<Match>> = ArrayList(phraseMatchesWitnessOrder)
        // The outputed list of matches that have been detected as transpositions.
        val transpositions: MutableList<List<Match>> = ArrayList()

        /*
         * loop here until the maximum distance == 0
         */
        while (true) {
            // Map 2
            val phraseMatchToDistanceMap: MutableMap<List<Match>, Int> = LinkedHashMap()
            // Calculate the diastance between the index of the witnesses and the graphs for each existing match list
            // Place the distances into the phraseMatchToDistanceMap array. 
            for (i in nonTransposedPhraseMatches.indices) {
                val graphIndex = phraseMatchesGraphIndex[i]
                val witnessIndex = phraseMatchesWitnessIndex[i]
                val distance = abs(graphIndex!! - witnessIndex!!)
                val phraseMatch = nonTransposedPhraseMatches[i]
                phraseMatchToDistanceMap[phraseMatch] = distance
            }
            // End function if the maximum calculated distance is 0 or list is empty 
            val distanceList: List<Int> = ArrayList(phraseMatchToDistanceMap.values)
            // DEBUG
            // println(distanceList)
            if (distanceList.isEmpty() || Collections.max(distanceList) == 0) {
                break
            }

            // sort phrase matches on distance, size
            // TODO: order by 3) graph rank?
            // TODO: I have not yet found evidence/a use case that
            // TODO: indicates that it is needed.
            val comp2 = Comparator { pm1: List<Match>, pm2: List<Match> ->
                // first order by distance
                val distance1 = phraseMatchToDistanceMap[pm1]!! // See if you 
                val distance2 = phraseMatchToDistanceMap[pm2]!!
                val difference = distance2 - distance1
                when {
                    difference != 0 -> difference
                    // Function determine which hase the most characters so as to determine which block will move
                    // The returned value is used to calculate by how much blocks must move
                    else -> determineSize(pm1) - determineSize(pm2)
                }
            }
            // Phrase matches sorted by there distance calculation. (This is where the actual transposition detection happens.)
            val sortedPhraseMatches: MutableList<List<Match>> = ArrayList(nonTransposedPhraseMatches.sortedWith(comp2))
            // Remove the phrase with smallest distance value
            val transposedPhrase: List<Match> = sortedPhraseMatches.removeAt(0)
            // Get graph idx of phrase with smallest distance
            val transposedIndex = phraseMatchToGraphIndex[transposedPhrase]
            // Get the index in phraseMatchesGraphIndex of previous index
            val graphIndex = phraseMatchesGraphIndex.indexOf(transposedIndex)
            // Get the witness index that is at the same position as the transposedIndex in graph index list
            val transposedWithIndex = phraseMatchesWitnessIndex[graphIndex]
            // Get matches that are at the same position as the original graph index
            val linkedTransposedPhrase = phraseMatchesGraphOrder[transposedWithIndex!!]
            // Updates all necessary arrays for the loop to pass to the next step.
            addTransposition(phraseMatchToGraphIndex, phraseMatchesWitnessIndex, phraseMatchesGraphIndex, nonTransposedPhraseMatches, transpositions, transposedPhrase)
            // Get the distance associated with the phrase identified as being transposed
            val distance = phraseMatchToDistanceMap[transposedPhrase]
            /*
             * If the distance of the transposed phrase is the same as the linked phrase in the graph AND that distance is greater than 1
             * Also add the linked phase to the detected transpositions
             * Only the last parameter changes in this case
             */
            if (distance == phraseMatchToDistanceMap[linkedTransposedPhrase] && distance!! > 1) {
                addTransposition(phraseMatchToGraphIndex, phraseMatchesWitnessIndex, phraseMatchesGraphIndex, nonTransposedPhraseMatches, transpositions, linkedTransposedPhrase)
            }
        }
        return transpositions
    }

    /**
     * Once all array and indexes have been selected this function takes all the relevant information and updates the necessary arrays for the loop to pass to the next step. 
     *
     * @param phraseMatchToIndex Dictionary that links matches to there index in the graph
     * @param phraseWitnessRanks List containing the matche indexes for witness lists (should be simple list from 0 to length(nb of witnesses))
     * @param phraseGraphRanks List containing the indexes of the matches in the graph but ordered by there presence in the witness match list
     * @param nonTransposedPhraseMatches List of mach lists in witness order.
     * @param transpositions The array that contains the detected transpositions. Is the array that this function will seek to fill.
     * @param transposedPhrase The phrase that has been identified as the transposition
     */
    private fun addTransposition(phraseMatchToIndex: Map<List<Match>, Int>, phraseWitnessRanks: MutableList<Int?>, phraseGraphRanks: MutableList<Int?>, nonTransposedPhraseMatches: MutableList<List<Match>>, transpositions: MutableList<List<Match>>, transposedPhrase: List<Match>) {
        val indexToRemove = phraseMatchToIndex[transposedPhrase]
        nonTransposedPhraseMatches.remove(transposedPhrase)
        transpositions.add(transposedPhrase)
        phraseGraphRanks.remove(indexToRemove)
        phraseWitnessRanks.remove(indexToRemove)
    }

    /*
     * Runs through the first dimension of phraseMatches and adds the vertex of it's
     * first element to matchedVertices. 
     * It then returns a ranking of the variant graph but only for the matchedVertices.
     */
    private fun rankTheGraph(phraseMatches: List<List<Match>>, base: VariantGraph): VariantGraphRanking {
        // rank the variant graph
        val matchedVertices: MutableSet<VariantGraph.Vertex> = HashSet()
        for (phraseMatch in phraseMatches) {
            matchedVertices.add(phraseMatch[0].vertex)
        }
        return VariantGraphRanking.ofOnlyCertainVertices(base, matchedVertices)
    }

    /*
     * in case of an a, b / b, a transposition we have to determine whether a or b
     * stays put. the phrase with the most character stays still if the tokens are
     * not simple tokens the phrase with the most tokens stays put
     */
    private fun determineSize(t: List<Match>): Int {
        val firstMatch = t[0]
        if (firstMatch.token !is SimpleToken) {
            return t.size
        }
        var charLength = 0
        for (m in t) {
            val token = m.token as SimpleToken
            charLength += token.normalized.length
        }
        return charLength
    }
}