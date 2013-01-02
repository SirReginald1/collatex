package eu.interedition.collatex.matching;

import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import eu.interedition.collatex.AbstractTest;
import eu.interedition.collatex.Token;
import eu.interedition.collatex.VariantGraph;
import eu.interedition.collatex.simple.SimpleWitness;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class NearMatcherTest extends AbstractTest {
  
  @Test
  public void nearTokenMatching() {
    final SimpleWitness[] w = createWitnesses("near matching yeah", "nar matching");
    final VariantGraph graph = collate(w[0]);
    final ListMultimap<Token, VariantGraph.Vertex> matches = Matches.between(graph.vertices(), w[1].getTokens(), new EditDistanceTokenComparator()).getAll();

    assertEquals(2, matches.size());
    assertEquals(w[0].getTokens().get(0), Iterables.getFirst(Iterables.get(matches.get(w[1].getTokens().get(0)), 0).tokens(), null));
    assertEquals(w[0].getTokens().get(1), Iterables.getFirst(Iterables.get(matches.get(w[1].getTokens().get(1)), 0).tokens(), null));
  }
}
