#This will parse and plot the log file
import numpy as np 
import matplotlib.pyplot as plt

def create_log_graphs(log_file, graph_path):
	"""
	log_file: filepath of log data
	"""
	with open(log_file, "r") as f:
		data = [j for i in f.readlines() for j in i.replace("\t", "").replace("\n","").split(",")]

	step = [float(i.replace("step: ","")) for i in data if "step" in i]
	total_loss = [float(i.replace("total loss: ","")) for i in data if "total loss" in i]
	Regularization_Loss = [float(i.replace("Regularization loss: ","")) for i in data if "Regularization loss" in i]
	Reconstruction_loss = [float(i.replace("Reconstruction loss: ","")) for i in data if "Reconstruction loss" in i]
	kl_weight= [float(i.replace("kl weight: ","").replace(",","")) for i in data if "kl weight" in i]
	average_stddev = [float(i.replace("average stddev ","")) for i in data if "average stddev" in i]
	stddev_lower_bound = [float(data[i].replace("stddev range [","").split(", ")[0]) for i in range(len(data)) if "stddev range" in data[i]]
	stddev_upper_bound = [float(data[i+1].replace("]","")) for i in range(len(data)) if "stddev range" in data[i]]
	average_mean = [float(i.replace("average mean ","")) for i in data if "average mean" in i]
	mean_lower_bound = [float(data[i].replace("mean range [","").split(", ")[0]) for i in range(len(data)) if "mean range" in data[i]]
	mean_upper_bound = [float(data[i+1].replace("]","")) for i in range(len(data)) if "mean range" in data[i]]



	plt.clf()
	subplot = [4,2]
	f, axarr = plt.subplots(*subplot, constrained_layout=True,figsize=(10,5*4))
	captions = []
	plots = [
		total_loss, 
		Regularization_Loss, 
		Reconstruction_loss, 
		kl_weight, 
		average_stddev, 
		[
		stddev_upper_bound,
		stddev_lower_bound, 
		], 
		average_mean, 
		[
		mean_upper_bound,
		mean_lower_bound, 
		]
	]

	captions = [
		"total loss", 
		"Regularization Loss", 
		"Reconstruction loss", 
		"kl weight", 
		"average stddev", 
		[
		"stddev upper bound",
		"stddev lower bound", 
		], 
		"average mean", 
		[
		"mean upper bound",
		"mean lower bound", 
		]
	]

	for i in range(len(plots)):
		if type(captions[i]) == str:
			axarr[i//2, i%2].set_title(captions[i])	
			plot = [plots[i]]
			caption = [captions[i]]
		else:
			plot=plots[i]
			caption = captions[i]

		for j in range(len(caption)):
			axarr[i//2, i%2].plot(step, plot[j], label=caption[j])
			if j >0:
				axarr[i//2, i%2].legend()
	f.show()
	plt.savefig(graph_path)
	plt.close()