import os
import shutil
from make_gif import make_gif
from plot_log import create_log_graphs
main_path = "predictions"
dest_path = "images"
folders = os.listdir(main_path)
#print(folders)
for model_path in folders:
	folders_path = os.path.join(main_path, model_path)
	images_path = os.path.join(folders_path, "images/")
	if not os.path.exists(images_path):
		continue

	log_file = os.path.join(folders_path, "log.txt")
	with open(".gitignore") as f:
		current_path = images_path+"\n"
		all_lines = f.readlines()
		if not current_path in all_lines:
			f.write(current_path)
	graph_path = os.path.join(dest_path, "%s_graphs.jpg"%model_path)
	gif_path = os.path.join(dest_path, "%s.gif"%model_path)
	create_log_graphs(log_file, graph_path)
	make_gif(gif_path, images_path)

	images = [i for i in os.listdir(images_path) if i.endswith(".jpg")]
	images = [i for _,i in sorted(zip([int(i.split("_")[-1].split(".")[0]) for i in images], images))]

	images = [i for i in os.listdir(images_path) if i.endswith(".jpg")]
	final_result = os.path.join(images_path, [i for _,i in sorted(zip([int(i.split("_")[-1].split(".")[0]) for i in images], images))][-1])
	new_latest_result_path = os.path.join(dest_path, "%s.jpg"%model_path)

	shutil.copyfile(final_result, new_latest_result_path)