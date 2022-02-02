OSLO: Open Source framework for Large-scale transformer Optimization
=====================================================================

.. container::

      .. image:: https://github.com/tunib-ai/oslo/raw/master/assets/oslo.png

| 
|

What’s New:
===========

-  February 02, 2022 `Add activation checkpointing
   <https://github.com/tunib-ai/oslo/releases/tag/v2.0.0a1>`__.
-  January 30, 2022 `Released OSLO 2.0 alpha
   version <https://github.com/tunib-ai/oslo/releases/tag/v2.0.0a0>`__.
-  December 30, 2021 `Add Deployment
   Launcher <https://github.com/tunib-ai/oslo/releases/tag/v1.0>`__.
-  December 21, 2021 `Released OSLO
   1.0 <https://github.com/tunib-ai/oslo/releases/tag/v1.0>`__.

What is OSLO about?
===================

OSLO is a framework that provides various GPU based optimization
technologies for large-scale modeling. 3D Parallelism and Kernel Fusion
which could be useful when training a large model like
`EleutherAI/gpt-j-6B <https://huggingface.co/EleutherAI/gpt-j-6B>`__ are
the key features. OSLO makes these technologies easy-to-use by magical
compatibility with `Hugging Face
Transformers <https://github.com/huggingface/transformers>`__ that is
being considered as a de facto standard in 2021.

Installation
============

OSLO can be easily installed using the pip package manager. All the
dependencies such as `torch <https://pypi.org/project/torch/>`__ and
`transformers <https://pypi.org/project/transformers/>`__ should be installed
automatically with the following command. Be careful that the ‘core’ is
in the PyPI project name.

.. code:: console

   pip install oslo-core


NOTE: OSLO 2.0.0 is still an alpha version, so you must specify the version.

.. code:: console

   pip install oslo-core==2.0.0a2

Basic Usage
====================

It only takes a single line of code. Now feel free to train and infer a large transformer model. 😎

.. code::

  import oslo

  model = oslo.initialize(model, "oslo-config.json")

Documents
====================

.. toctree::
   :maxdepth: 1
   :caption: CONFIGURATION

   CONFIGURATION/model_parallelism
   CONFIGURATION/activation_checkpointing

.. toctree::
   :maxdepth: 1
   :caption: TUTORIALS

   TUTORIALS/tensor_model_parallelism
   TUTORIALS/tensor_model_data_parallelism
   TUTORIALS/tensor_model_zero_data_parallelism
   TUTORIALS/activation_checkpointing

Administrative Notes
====================

Citing OSLO
-----------

If you find our work useful, please consider citing:

::

   @misc{oslo,
     author       = {Ko, Hyunwoong and Kim, Soohwan and Park, Kyubyong},
     title        = {OSLO: Open Source framework for Large-scale transformer Optimization},
     howpublished = {\url{https://github.com/tunib-ai/oslo}},
     year         = {2021},
   }

Licensing
---------

The Code of the OSLO project is licensed under the terms of the `Apache
License 2.0 <LICENSE.apache-2.0>`__.

Copyright 2021 TUNiB Inc. http://www.tunib.ai All Rights Reserved.

Acknowledgements
----------------

The OSLO project is built with GPU support from the `AICA (Artificial
Intelligence Industry Cluster Agency) <http://www.aica-gj.kr>`__.
