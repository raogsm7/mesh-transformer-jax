# Fine Tune Mesh Transformer JAX
  

1. Loginto your google drive and create directories for 
    * MyDrive/colab_data/ckpt_dir
    * MyDrive/colab_data/finetuned_ckpt_dir

2. Upload checkpoint data to google drive or g drive
    - MyDrive/colab_data/ckpt_dir/step_0/shard_0.....

3. Upload tfrecords of your data to gdrive or map local drive to gdrive. Following drive mapped to g-drive:
- <"your computername">/openwebtext_tokenized/<"all ttf recordes">

4. Open GPT-J-6B_Train_v1.ipynb from https://colab.research.google.com/drive/1u4aJcexh2ONBYG97XdXLWr79FArwEl85?usp=sharing 


5. Mount your drive by running following code of ipynb file

Mount your drive
```
from google.colab import drive
 drive.mount('/content/drive')
```
System will provide you with link for API key, click the link and login with your credentials and input the key. Your google drive is now mounted

6. Run the cell that have !git clone ...... 
- It loads the the repo from github
- Runs requirements file
- installs jax and tensorflow


7. Open index file for train and val

- Locate the index file for train at "/content/mesh-transformer-jax/data/openwebtext.train.index"
- update the file with path for uploaded tf records 
- Similarly update the index file for val at "/content/mesh-transformer-jax/data/openwebtext.val.index"

8. Update config file paramters as per TPU size and Cloud or Gdrive 

- Config file for finetune is at "mesh-transformer-jax/configs/6B_roto_256_ft.json


```
  "layers": 3,
  "d_model": 512,
  "n_heads": 8,
  "n_vocab": 50400,
  "norm": "layernorm",
  "pe": "rotary",
  "pe_rotary_dims": 64, (RoPE dimensions)

  "seq": 256,
  "cores_per_replica": 1,
  "per_replica_batch": 1,
  "gradient_accumulation_steps": 5,

  "warmup_steps": 3,
  "anneal_steps": 3,
  "lr": 1.2e-4,
  "end_lr": 1.2e-5,
  "weight_decay": 0.1,
  "total_steps":10,

  "tpu_size": 2,
```

Update model directiry as per your gdrive path in this config file - 
"model_dir": "/content/drive/MyDrive/colab_data/finetuned_ckpt_dir",

9. Keep executing cells one by one by replacing path for replace path for ckpt_dir of your g-drive. In finetune cell here, there are three options given

Option 1 - Finetune with pre-existing checkpoints, 

```
!python3 /content/mesh-transformer-jax/train_ft_6jb.py --config=/content/mesh-transformer-jax/configs/6B_roto_256_ft.json --tune-model-path=/content/drive/MyDrive/colab_data/ckpt_dir/
```

Option 2 - Finetune without any pre-existing checkpoings. In this case finetune works as train program


```
!python3 /content/mesh-transformer-jax/train_ft_6jb.py --config=/content/mesh-transformer-jax/configs/6B_roto_256_ft.json 
```

Option 3 - kept it for testing new program parameters (please do not run this)

```
!python3 /content/mesh-transformer-jax/train_ft_6jb.py --config=/content/mesh-transformer-jax/configs/6B_roto_256_ft.json --tune-model-path=/content/mesh-transformer-jax/data/ckpt_dir/ --fresh-opt=True
```


10. While runnig the finetune cell, the program will ask for wandb credentials, use your existing account option. If you do not have account create new account following the instructions given on the cell output. Enter your wandb key to continue.

11. Run the last cell, to unmount and save the folder to gdrive