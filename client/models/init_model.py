from autocoder import create_seed_model
import numpy as np

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = create_seed_model()
	outfile_name = "../../seed/seed.npz"

	weights = model.get_weights()

	weights_dict = {}
	for i, w in enumerate(weights):
		weights_dict[str(i)] = w

	np.savez_compressed(outfile_name, **weights_dict)