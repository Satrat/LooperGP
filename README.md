- full-data-config_5_lat1024.yml
	
	main config file with all the parameters for generation and training

- fulldataset-song-artist-train_data_XL.npz

	dataset in numpy format, for training

- inference_fd5_lat1024.py
	
	script for generation, dependent on:
		- main config file
		- model weights
		- dataset vocabulary and reverse vocabulary

- model_ead.py

	script for model backbone, adapted from this repo:
	https://github.com/YatingMusic/compound-word-transformer

- model weights

	folder containing:
		- model weights (from epoch 200, the best one we have)
		- a config file (mostly useless, but necessary for it to run; can be ignored)

- modules.py

	script for model backbone, from:
	https://github.com/YatingMusic/compound-word-transformer

- rev_vocab_song_artist.pkl

	reverse vocabulary, necessary for training/inference

- train_randomsampling_5_lat1024.py

	script for training, adapted from:
	https://github.com/YatingMusic/compound-word-transformer

	dependent on:
		- main config file
		- dataset vocabulary and reverse vocabulary
		- dataset in numpy format

- vocab_song_artist.pkl

	vocabulary for the dataset, necessary for training/inference