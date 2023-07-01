def mySqrt(self, x: int) -> int:
  '''Given a non-negative integer x,
  return the square root of x rounded down to the nearest integer.
  The returned integer should be non-negative as well.
  '''
  if x == 0:
      return 0
  guess = x
  while True:
      new_guess = (guess + x // guess) // 2
      if new_guess >= guess:
          return guess
      guess = new_guess
