from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")

slices = [59219,55466,47544,36443,35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'PYTHON', 'JAVA']
explode = [0,0,0,0.1,0]

plt.pie(slices, labels=labels, explode=explode, shadow = True, wedgeprops={'edgecolor':'black'}, autopct='%1.1f%%')

plt.title("Pie chart for programming languages")
plt.tight_layout()
plt.show()