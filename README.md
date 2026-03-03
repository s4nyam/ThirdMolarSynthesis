# Synthesizing Third Molars with Denoising Diffusion Probabilistic Models and Generative Adversarial Networks

[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](https://arxiv.org/abs/2507.09227v1)
[![arXiv](https://img.shields.io/badge/highquality-paper-pink)](https://drive.google.com/file/d/1iRBodVYvZ93cZXr_cXxhw19ELZE6OPCQ/view?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![pyngrok](https://img.shields.io/badge/PyNgRok-7.2%2B-yellow)](https://pypi.org/project/pyngrok/)


> 📢 Official PyTorch implementation of the **Third Molar and Mandibular Canal Synthesis**:  
> **Synthesizing Third Molars with Denoising Diffusion Probabilistic Models and Generative Adversarial Networks**  
> Sanyam Jain, Sara Haghighat, Johan Andreas Balle Rubak, Mostafa Aldesoki, Akhilanand Chaurasia, Sarah Sadat Ehsani, Faezeh Dehghan Ghanatkaman, Ahmad Badruddin Ghazali, Julien Issa, Basel Khalil, Rishi Ramani, Ruben Pauwels
> [[Project]](https://github.com/s4nyam/ThirdMolarSynthesis) | [[Code]](https://github.com/s4nyam/ThirdMolarSynthesis)

---
## 🌟 Highlights

- **Dual Generative Framework:** We investigate both **Denoising Diffusion Probabilistic Models (DDPM/DDIM)** and **Generative Adversarial Networks (GANs)** in unconditional and class-conditional settings for third molar (3M) region-of-interest (ROI) synthesis.

- **Multi-Dataset Curation:** Five publicly available panoramic radiograph datasets were harmonized through a unified preprocessing pipeline, including resizing, annotation scaling, cropping, padding, and spatial standardization, resulting in 5,416 expert-curated ROIs.

- **Architecture-Level Innovation:** We introduce a **Spatially-Aware Excitation (SAE)** module within a modified FastGAN framework to improve anatomical plausibility by preserving spatially critical structures such as the mandibular canal and root apex.

- **Conditional Diffusion Modeling:** A label-conditioned diffusion model is implemented using timestep-aware embeddings and FiLM-based modulation to enable controlled synthesis across seven clinically defined anatomical classes.

- **Quantitative & Human Evaluation:** Synthetic realism is assessed using **Fréchet Inception Distance (FID)** and **Inception Score (IS)**, complemented by a structured human observer study in which eight dentists evaluated paired GAN and diffusion samples in a time-constrained real-vs-fake setting.

- **Clinical-Oriented Analysis:** Inter-observer agreement (Fleiss’ κ and pairwise Cohen’s κ) was analyzed to understand perceptual variability in anatomical realism assessment.

<p align="center">
  <img width="900" height="600" alt="Conditional third molar ROIs"
       src="https://github.com/user-attachments/assets/d0f29a15-bcd3-42d6-8bd9-2de2c06a1f16" />
  <br>
  <em>
    Figure: Conditional third molar ROIs for training (original), cGAN and cDiffusion models.
    Clearly, the lower-quality synthesis of cGAN compared with cDiffusion can be observed.
    Although cDiffusion does not yet produce high-quality images, it demonstrates clear improvement over cGAN.
  </em>
</p>


<p align="center">
  <img width="600" height="626" alt="Unconditional third molar ROIs"
       src="https://github.com/user-attachments/assets/b16f97e1-2898-411a-9113-7dc51a340fcd" />
  <br>
  <em>
    Figure: Unconditional third molar ROIs from the training dataset, GAN, and diffusion models.
    No substantial perceptual differences are observed between GAN and diffusion synthesis
    at the visual inspection level.
  </em>
</p>

## 📊 Dataset

<p align="center">
  <img src="https://dl3.pushbulletusercontent.com/fNlF9Ytp2g0cGoIaGoekrZmB550SMKZh/image.png" alt="T sne" width="500"/>
</p>

T-distributed stochastic neighbor embedding (t-SNE) plot of 500 random images picked from each source dataset. (You can study more about dataset here - https://arxiv.org/pdf/2507.09227)

**Table:** Public panoramic radiograph (PR) datasets used to curate the third molar (3M)–mandibular canal (MC) region-of-interest (ROI) dataset in this study. Only images with available third molar annotations were retained. All datasets were harmonized through a unified preprocessing pipeline including resizing, annotation scaling, standardized cropping, padding, and spatial alignment. * indicates that the dataset was later expanded, but only the originally available subset was used. ~ denotes variability in native image resolution within the dataset.

| Abbr.    | Images | Format | Availability | Year | Country | Resolution |
|---|---|---|---|---|---|---|
| [ADLD](https://www.kaggle.com/datasets/zwbzwb12341234/a-dual-labeled-dataset) | 500* | png | Kaggle | 2024 | China | ~2940×1435 or ~987×478 |
| [DENTEX](https://zenodo.org/records/7812323#.ZDQE1uxBwUG) | 3903 | png | Zenodo | 2023 | Switzerland | ~2950×1316 or ~1976×976 |
| [TSXK](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images) | 1196 | png | Kaggle | 2023 | DR Congo | 2041×1024 |
| [TUFTS](https://tdd.ece.tufts.edu/) | 1000 | jpg | On Request | 2022 | USA | 1615×840 |
| [USPFORP](https://pubmed.ncbi.nlm.nih.gov/38632036/) | 936 | jpg | On Request | 2024 | Brazil | 2903×1536 |

To use this table in your reseach, you can cite using: 

```bibtex
@misc{jain2025panodiffsrsynthesizingdentalpanoramic,
      title={PanoDiff-SR: Synthesizing Dental Panoramic Radiographs using Diffusion and Super-resolution}, 
      author={Sanyam Jain and Bruna Neves de Freitas and Andreas Basse-OConnor and Alexandros Iosifidis and Ruben Pauwels},
      year={2025},
      eprint={2507.09227},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2507.09227}, 
}
```
---


## Acknowledgments

- Gratitude to the open-source community for datasets and tools.
- Computing resources were supported by ECE and MaLeCi Aarhus University, Denmark. LUMI supercomputers were also used during the experimentation and ablation studies.

**Sanyam Jain<sup>1,10*</sup>, Sara Haghighat<sup>1,10</sup>, Johan Andreas Balle Rubak<sup>1</sup>,  
Mostafa Aldesoki<sup>2,10</sup>, Akhilanand Chaurasia<sup>3,10</sup>,  
Sarah Sadat Ehsani<sup>4,10</sup>, Faezeh Dehghan Ghanatkaman<sup>5,10</sup>,  
Ahmad Badruddin Ghazali<sup>6,10</sup>, Julien Issa<sup>7,10</sup>,  
Basel Khalil<sup>8,10</sup>, Rishi Ramani<sup>9,10</sup>,  
Ruben Pauwels<sup>1,10</sup>**

---

<sup>1</sup> Department of Dentistry and Oral Health, Aarhus University, Aarhus, Denmark  
<sup>2</sup> Private Dental Practice Dr. Jörg Deike, Hannover, Germany  
<sup>3</sup> Department of Oral Medicine and Radiology, King George’s Medical University, Lucknow, India  
<sup>4</sup> Department of Diagnosis and Oral Health, University of Louisville School of Dentistry, Louisville, USA  
<sup>5</sup> Faculty of Dentistry, Islamic Azad University of Medical Sciences, Tehran, Iran  
<sup>6</sup> Department of Oral Maxillofacial Surgery & Oral Diagnosis, Kulliyyah of Dentistry, International Islamic University Malaysia  
<sup>7</sup> Department of Oral Radiology & Digital Dentistry, Academic Centre for Dentistry Amsterdam (ACTA), University of Amsterdam and Vrije Universiteit Amsterdam, Amsterdam, The Netherlands  
<sup>8</sup> Private Practice, Stockholm, Sweden  
<sup>9</sup> Melbourne Dental School, University of Melbourne, Victoria, Australia  
<sup>10</sup> ITU/WHO/WIPO Global Initiative on Artificial Intelligence for Health, Topic Group Oral Health, Geneva, Switzerland  

📧 **Correspondence:** ruben.pauwels@dent.au.dk

---

## More results


<p align="center">
<img width="1562" height="941" alt="image" src="https://github.com/user-attachments/assets/2d6a4abe-b1a9-488d-ac79-57ad6deeb5f5" />
  <br>
  <em>
    Figure: 60 samples synthesized using GAN Model.
  </em>
</p>



<p align="center">
<img width="1557" height="942" alt="image" src="https://github.com/user-attachments/assets/3c0c68b9-0110-4045-8a3a-2f6773b15257" />
  <br>
  <em>
    Figure: 60 samples synthesized using Diffusion Model.
  </em>
</p>


## License

