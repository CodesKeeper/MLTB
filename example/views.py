from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
import json
from django.http import HttpResponse, JsonResponse
from theory.simulation_data import simulation_data
from sklearn.datasets.samples_generator import make_regression, make_classification, make_blobs
import numpy as np
from scipy import io as spio
from theory.kNearestNeighbor import KNN


def decision_tree(request):
    return render(request, 'algorithm/example/decision_tree.html')


def deep_learning(request):
    return render(request, 'algorithm/example/deep_learning.html')


def ensemble_learning(request):
    return render(request, 'algorithm/example/ensemble_learning.html')


def k_means(request):
    return render(request, 'algorithm/example/k_means.html')


def k_nearest_neighbor(request):
    return render(request, 'algorithm/example/k_nearest_neighbor.html')


def linear_regression(request):
    return render(request, 'algorithm/example/linear_regression.html')


def logistic_regression(request):
    return render(request, 'algorithm/example/logistic_regression.html')


def naive_bayes(request):
    return render(request, 'algorithm/example/naive_bayes.html')


def neural_network(request):
    return render(request, 'algorithm/example/neural_network.html')


def support_vector_machine(request):
    return render(request, 'algorithm/example/support_vector_machine.html')
