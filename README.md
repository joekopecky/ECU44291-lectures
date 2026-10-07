# ECU44291 Quantitative Macroeconomics: lecture notebooks and code

Trinity College Dublin, Michaelmas Term 2026. Joseph Kopecky.

One folder per week. Before each lecture the blank notebook is posted; after it, the complete notebook and the week's code. The blank notebooks open in Google Colab with nothing to install.

| Week | Blank notebook | Open in Colab | Complete notebook and code |
|---|---|---|---|
| 1 | [lecture01_solow_blank.ipynb](week01/lecture01_solow_blank.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joekopecky/ECU44291-lectures/blob/main/week01/lecture01_solow_blank.ipynb) | [lecture01_solow.ipynb](week01/lecture01_solow.ipynb), [code](week01/code) |
| 2 | [lecture02_rootfinding_olg_blank.ipynb](week02/lecture02_rootfinding_olg_blank.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joekopecky/ECU44291-lectures/blob/main/week02/lecture02_rootfinding_olg_blank.ipynb) | [lecture02_rootfinding_olg.ipynb](week02/lecture02_rootfinding_olg.ipynb), [code](week02/code) |
| 3 | [lecture03_lifecycle_blank.ipynb](week03/lecture03_lifecycle_blank.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joekopecky/ECU44291-lectures/blob/main/week03/lecture03_lifecycle_blank.ipynb) | [lecture03_lifecycle.ipynb](week03/lecture03_lifecycle.ipynb), [code](week03/code) |
| 4 | [lecture04_vfi_blank.ipynb](week04/lecture04_vfi_blank.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joekopecky/ECU44291-lectures/blob/main/week04/lecture04_vfi_blank.ipynb) | after the lecture |

## Tutorials

| Tutorial | Exercise notebook | Worked | Open the worked version in Colab |
|---|---|---|---|
| 1 (7 Oct) | [tutorial01.ipynb](week03/tutorial01.ipynb) | [tutorial01_complete.ipynb](week03/tutorial01_complete.ipynb); Exercise 3's broken solvers in [tutorial01_debrief.ipynb](week03/tutorial01_debrief.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joekopecky/ECU44291-lectures/blob/main/week03/tutorial01_complete.ipynb) |

Slides, method cards, deadlines and the module's rules on generative AI are on Blackboard. Your own project code lives in your private repository, not here.

## Getting the notebooks into your project

Clone this repository once, next to your project, and pull it each week:

```
cd ~/code
git clone https://github.com/joekopecky/ECU44291-lectures.git   # once
cd ECU44291-lectures
git pull                                                        # every week
```

Then copy the week's blank notebook into your project and work there:

```
cp ~/code/ECU44291-lectures/week02/lecture02_rootfinding_olg_blank.ipynb ~/code/ECU44291-project/notebooks/
```

Never edit files inside this folder; `git pull` will refuse if you have.
