# ISSUES

1. [ ] Parsing guard conditions in match statements fails

    `[*, d] where d > 512: Linear(d, 512) -> out` => Parse error: Expected Colon, found Gt
          `^_____________^`

2. [ ] Debugging sucks, error messages aren't descriptive, doesn't show offending code
