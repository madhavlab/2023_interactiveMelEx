# **Interactive singing melody extraction based on active adaptation**

The paper associated with these codes:
- Kavya Ranjan Saxena and Vipul Arora. "Interactive singing melody extraction based on active adaptation." IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.

Trained models are in the folder `weights`

Step-by-step training: <br /> <br />
**STEP 1:** Run the **pretrain_basemodel.py** file to pre-train the melody estimation model. The base model is present in the models.py file. <br />
**STEP 2:** Run the **pretrain_confmodel.py** file to pre-train the confidence model. The confidence model is present in the models.py file. <br />
**STEP 3:** Once we have obtained the pre-train base and confidence model, we apply active-meta-learning by running the **active_meta_training.py**file. <br />
**STEP 4:** Once we have obtained the active-meta-trained model, we use it to adapt to the audios in the target domain by running the **active_meta_testing.py** file <br />

## Citation
If you use this work, please cite us
```
  @article{saxena2024interactive,
  title={Interactive singing melody extraction based on active adaptation},
  author={Saxena, Kavya Ranjan and Arora, Vipul},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2024},
  publisher={IEEE}
}
```
