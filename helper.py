import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.bar(range(len(scores)), scores, alpha=0.6, label='Score')
    plt.plot(mean_scores, color='orange', label='Mean Score')
    plt.legend(loc='upper left')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], f'{scores[-1]} ({max(scores)})')
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.01)
