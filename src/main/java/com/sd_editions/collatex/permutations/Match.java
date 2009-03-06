package com.sd_editions.collatex.permutations;

public class Match implements Comparable<Match> {
  private final Word word1;
  private final Word word2;
  public final float levenshteinDistance;

  public Match(Word baseWord, Word witnessWord) {
    this(baseWord, witnessWord, 0);
  }

  public Match(Word baseWord, Word witnessWord, float levDistance) {
    this.word1 = baseWord;
    this.word2 = witnessWord;
    this.levenshteinDistance = levDistance;
  }

  @Override
  public String toString() {
    return "(" + word1.position + "->" + word2.position + ")";
  }

  public Word getWitnessWord() {
    return word2;
  }

  public Word getBaseWord() {
    return word1;
  }

  public boolean equals(Match other) {
    return this.word1.equals(other.word1) && this.word2.equals(other.word2);
  }

  @Override
  public int hashCode() {
    return word1.hashCode() + word2.hashCode();
  }

  public int compareTo(Match m2) {
    return getBaseWord().position - m2.getBaseWord().position;
  }
}
