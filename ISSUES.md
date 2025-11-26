# ISSUES

1. [x] Parsing guard conditions in match statements fails

    `[*, d] where d > 512: Linear(d, 512) -> out` => Parse error: Expected Colon, found Gt
    Guard statements parsing now

2. [x] Debugging sucks, error messages aren't descriptive, doesn't show offending code
   
    miette installed and working!
    
