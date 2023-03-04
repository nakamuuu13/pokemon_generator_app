import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import cv2

import openai

from flask import Flask, request, render_template