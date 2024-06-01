
from rouge import Rouge 

hypothesis = """
  Anna searches for her favorite book 'The Silent Stars', and Ben admits 
  he borrowed it without asking, leading to a plan to discuss it later
"""
reference = """
  Anna realizes Ben borrowed her favorite book without asking, and they 
  agree to discuss it once he's finished reading
"""

hypothesis = 'APPIOT\n</s>'
reference = 'APPIOT'

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)

print(scores)