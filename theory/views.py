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
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def index(request):
    return render(request, 'index.html')


def user_login(request):
    if request.method == "POST":
        user = authenticate(request, username=request.POST['nick_name'], password=request.POST['password'])
        if user is None:
            return render(request, 'login.html', {'error': '用户名或密码错误'})
        else:
            login(request, user)
            return render(request, 'algorithm/majorization/least_square_method.html', {'user_name': user})
    elif request.method == "GET":
        return render(request, 'login.html')


def user_logout(request):
    logout(request)
    return redirect('index')


def register(request):
    if request.method == "POST":
        register_form = UserCreationForm(request.POST)
        if register_form.is_valid():
            register_form.save()
            user = authenticate(username=register_form.cleaned_data['username'],
                                password=register_form.cleaned_data['password1'])
            login(request, user)
            return render(request, 'index.html', {'user_name': user})
    elif request.method == 'GET':
        return render(request, 'register.html')
    else:
        register_form = UserCreationForm()
    message = {'register_form': register_form}
    return render(request, 'algorithm/majorization/least_square_method.html', message)


def about(request):
    return render(request, 'about.html')


def document(request):
    return render(request, 'document/document.html')


def least_square_method(request):
    if request.method == 'POST':
        K = request.POST.get('k')
        B = request.POST.get('b')
        N_true = request.POST.get('N_true')
        N_noise = request.POST.get('N_noise')
        mu = request.POST.get('mu')
        sigma = request.POST.get('sigma')
        sim = simulation_data(float(K), float(B), float(N_true), float(N_noise), float(mu), float(sigma))
        json_data = sim.turn_json('LSM')
        return JsonResponse(json_data, safe=False)
    elif request.method == 'GET':
        sim = simulation_data()
        json_data = sim.turn_json('LSM')
        return render(request, 'algorithm/majorization/least_square_method.html',
                      {'data': json.dumps(json_data)})


def batch_gradient_descent(request):
    if request.method == 'POST':
        K = request.POST.get('k')
        B = request.POST.get('b')
        N_true = request.POST.get('N_true')
        N_noise = request.POST.get('N_noise')
        mu = request.POST.get('mu')
        sigma = request.POST.get('sigma')
        sim = simulation_data(float(K), float(B), float(N_true), float(N_noise), float(mu), float(sigma))
        json_data = sim.turn_json('BGD')
        return JsonResponse(json_data, safe=False)
    elif request.method == 'GET':
        sim = simulation_data()
        json_data = sim.turn_json('BGD')
        return render(request, 'algorithm/majorization/batch_gradient_descent.html',
                      {'data': json.dumps(json_data)})


def stochastic_gradient_descent(request):
    if request.method == 'POST':
        K = request.POST.get('k')
        B = request.POST.get('b')
        N_true = request.POST.get('N_true')
        N_noise = request.POST.get('N_noise')
        mu = request.POST.get('mu')
        sigma = request.POST.get('sigma')
        sim = simulation_data(float(K), float(B), float(N_true), float(N_noise), float(mu), float(sigma))
        json_data = sim.turn_json('SGD')
        return JsonResponse(json_data, safe=False)
    elif request.method == 'GET':
        sim = simulation_data()
        json_data = sim.turn_json('SGD')
        return render(request, 'algorithm/majorization/stochastic_gradient_descent.html',
                      {'data': json.dumps(json_data)})


def linear_simulation_data(request):
    if request.method == 'POST':
        n_samples = request.POST.get('n_samples')
        n_features = request.POST.get('n_features')
        n_redundant = request.POST.get('n_redundant')
        n_classes = request.POST.get('n_classes')
        n_clusters_per_class = request.POST.get('n_clusters_per_class')
        #print(float(n_samples), float(n_features), float(n_redundant), float(n_classes), float(n_clusters_per_class))
        # 初始化参数--------------------
        n_samples = int(n_samples)
        n_features = int(n_features)
        n_redundant = int(n_redundant)
        n_clusters_per_class = int(n_clusters_per_class)
        n_classes = int(n_classes)
        # 仿真数据生成------------------
        X1, Y1 = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=n_redundant,
                                     n_clusters_per_class=n_clusters_per_class, n_classes=n_classes)

        # 仿真数据分组------------------
        def arr_turn(X1, Y1, n_classes, n_features):
            # 创建数组---------------
            def duplicate_checking_one(Y1, n_features):
                x = len(Y1)
                arr_x = np.zeros((x, n_features))
                return arr_x

            def duplicate_checking_two(Y1, n_features):
                x, y = 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        x += 1
                    elif Y1[i] == 1:
                        y += 1
                arr_x = np.zeros((x, n_features))
                arr_y = np.zeros((y, n_features))
                return arr_x, arr_y

            def duplicate_checking_three(Y1, n_features):
                x, y, z = 0, 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        x += 1
                    elif Y1[i] == 1:
                        y += 1
                    elif Y1[i] == 2:
                        z += 1
                arr_x = np.zeros((x, n_features))
                arr_y = np.zeros((y, n_features))
                arr_z = np.zeros((z, n_features))
                return arr_x, arr_y, arr_z,

            def duplicate_checking_four(Y1, n_features):
                x, y, z, q = 0, 0, 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        x += 1
                    elif Y1[i] == 1:
                        y += 1
                    elif Y1[i] == 2:
                        z += 1
                    elif Y1[i] == 3:
                        q += 1
                arr_x = np.zeros((x, n_features))
                arr_y = np.zeros((y, n_features))
                arr_z = np.zeros((z, n_features))
                arr_q = np.zeros((q, n_features))
                return arr_x, arr_y, arr_z, arr_q

            if n_classes == 1:
                arr_x = duplicate_checking_one(Y1, n_features)
                for i in range(0, len(Y1)):
                    arr_x[i] = X1[i]
                return arr_x
            elif n_classes == 2:
                arr_x, arr_y = duplicate_checking_two(Y1, n_features)
                j, k = 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        arr_x[j] = X1[i]
                        j += 1
                        continue
                    elif Y1[i] == 1:
                        arr_y[k] = X1[i]
                        k += 1
                return arr_x, arr_y
            elif n_classes == 3:
                arr_x, arr_y, arr_z = duplicate_checking_three(Y1, n_features)
                j, k, l = 0, 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        arr_x[j] = X1[i]
                        j += 1
                        continue
                    elif Y1[i] == 1:
                        arr_y[k] = X1[i]
                        k += 1
                        continue
                    elif Y1[i] == 2:
                        arr_z[l] = X1[i]
                        l += 1
                return arr_x, arr_y, arr_z
            elif n_classes == 4:
                arr_x, arr_y, arr_z, arr_q = duplicate_checking_four(Y1, n_features)
                j, k, l, m = 0, 0, 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        arr_x[j] = X1[i]
                        j += 1
                        continue
                    elif Y1[i] == 1:
                        arr_y[k] = X1[i]
                        k += 1
                        continue
                    elif Y1[i] == 2:
                        arr_z[l] = X1[i]
                        l += 1
                    elif Y1[i] == 3:
                        arr_q[m] = X1[i]
                        m += 1
                return arr_x, arr_y, arr_z, arr_q
            # ----------------------

        # ------------------------------

        json_data = {}
        if n_classes == 1:
            arr_x = arr_turn(X1, Y1, n_classes, n_features)
            json_data['data1'] = arr_x.tolist()
        elif n_classes == 2:
            arr_x, arr_y = arr_turn(X1, Y1, n_classes, n_features)
            json_data['data1'] = arr_x.tolist()
            json_data['data2'] = arr_y.tolist()
        elif n_classes == 3:
            arr_x, arr_y, arr_z = arr_turn(X1, Y1, n_classes, n_features)
            json_data['data1'] = arr_x.tolist()
            json_data['data2'] = arr_y.tolist()
            json_data['data3'] = arr_z.tolist()
        elif n_classes == 4:
            arr_x, arr_y, arr_z, arr_q = arr_turn(X1, Y1, n_classes, n_features)
            json_data['data1'] = arr_x.tolist()
            json_data['data2'] = arr_y.tolist()
            json_data['data3'] = arr_z.tolist()
            json_data['data4'] = arr_q.tolist()
        json_data['x'] = len(json_data)
        return JsonResponse(json_data, safe=False)
    elif request.method == 'GET':
        n_samples = 400
        n_features = 2
        n_redundant = 0
        n_clusters_per_class = 1
        n_classes = 3
        # 仿真数据生成------------------
        X1, Y1 = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=n_redundant,
                                     n_clusters_per_class=n_clusters_per_class, n_classes=n_classes)

        # 仿真数据分组------------------
        def arr_turn(X1, Y1, n_classes, n_features):
            # 创建数组---------------
            def duplicate_checking_one(Y1, n_features):
                x = len(Y1)
                arr_x = np.zeros((x, n_features))
                return arr_x

            def duplicate_checking_two(Y1, n_features):
                x, y = 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        x += 1
                    elif Y1[i] == 1:
                        y += 1
                arr_x = np.zeros((x, n_features))
                arr_y = np.zeros((y, n_features))
                return arr_x, arr_y

            def duplicate_checking_three(Y1, n_features):
                x, y, z = 0, 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        x += 1
                    elif Y1[i] == 1:
                        y += 1
                    elif Y1[i] == 2:
                        z += 1
                arr_x = np.zeros((x, n_features))
                arr_y = np.zeros((y, n_features))
                arr_z = np.zeros((z, n_features))
                return arr_x, arr_y, arr_z,

            def duplicate_checking_four(Y1, n_features):
                x, y, z, q = 0, 0, 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        x += 1
                    elif Y1[i] == 1:
                        y += 1
                    elif Y1[i] == 2:
                        z += 1
                    elif Y1[i] == 3:
                        q += 1
                arr_x = np.zeros((x, n_features))
                arr_y = np.zeros((y, n_features))
                arr_z = np.zeros((z, n_features))
                arr_q = np.zeros((q, n_features))
                return arr_x, arr_y, arr_z, arr_q

            if n_classes == 1:
                arr_x = duplicate_checking_one(Y1, n_features)
                for i in range(0, len(Y1)):
                    arr_x[i] = X1[i]
                return arr_x
            elif n_classes == 2:
                arr_x, arr_y = duplicate_checking_two(Y1, n_features)
                j, k = 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        arr_x[j] = X1[i]
                        j += 1
                        continue
                    elif Y1[i] == 1:
                        arr_y[k] = X1[i]
                        k += 1
                return arr_x, arr_y
            elif n_classes == 3:
                arr_x, arr_y, arr_z = duplicate_checking_three(Y1, n_features)
                j, k, l = 0, 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        arr_x[j] = X1[i]
                        j += 1
                        continue
                    elif Y1[i] == 1:
                        arr_y[k] = X1[i]
                        k += 1
                        continue
                    elif Y1[i] == 2:
                        arr_z[l] = X1[i]
                        l += 1
                return arr_x, arr_y, arr_z
            elif n_classes == 4:
                arr_x, arr_y, arr_z, arr_q = duplicate_checking_four(Y1, n_features)
                j, k, l, m = 0, 0, 0, 0
                for i in range(0, len(Y1)):
                    if Y1[i] == 0:
                        arr_x[j] = X1[i]
                        j += 1
                        continue
                    elif Y1[i] == 1:
                        arr_y[k] = X1[i]
                        k += 1
                        continue
                    elif Y1[i] == 2:
                        arr_z[l] = X1[i]
                        l += 1
                    elif Y1[i] == 3:
                        arr_q[m] = X1[i]
                        m += 1
                return arr_x, arr_y, arr_z, arr_q
            # ----------------------

        # ------------------------------

        json_data = {}
        if n_classes == 1:
            arr_x = arr_turn(X1, Y1, n_classes, n_features)
            json_data['data1'] = arr_x.tolist()
        elif n_classes == 2:
            arr_x, arr_y = arr_turn(X1, Y1, n_classes, n_features)
            json_data['data1'] = arr_x.tolist()
            json_data['data2'] = arr_y.tolist()
        elif n_classes == 3:
            arr_x, arr_y, arr_z = arr_turn(X1, Y1, n_classes, n_features)
            json_data['data1'] = arr_x.tolist()
            json_data['data2'] = arr_y.tolist()
            json_data['data3'] = arr_z.tolist()
        elif n_classes == 4:
            arr_x, arr_y, arr_z, arr_q = arr_turn(X1, Y1, n_classes, n_features)
            json_data['data1'] = arr_x.tolist()
            json_data['data2'] = arr_y.tolist()
            json_data['data3'] = arr_z.tolist()
            json_data['data4'] = arr_q.tolist()
        json_data['x'] = len(json_data)
        return render(request, 'algorithm/simulation/linear.html', {'data': json.dumps(json_data)})


def classify_simulation_data(request):
    return render(request, 'algorithm/simulation/classify.html')


def license_plate_recognition(request):
    return render(request, 'application/license_plate_recognition.html')


def cluster_simulation_data(request):
    return render(request, 'algorithm/simulation/cluster.html')


def k_nearest_neighbor(request):
    if request.method == 'POST':
        K = request.POST.get('k')
        B = request.POST.get('b')
        N_true = request.POST.get('N_true')
        N_noise = request.POST.get('N_noise')
        mu = request.POST.get('mu')
        sigma = request.POST.get('sigma')
        sim = simulation_data(float(K), float(B), float(N_true), float(N_noise), float(mu), float(sigma))
        json_data = sim.turn_json('BGD')
        return JsonResponse(json_data, safe=False)
    elif request.method == 'GET':
        json_data = KNN().make_data()
        return render(request, 'algorithm/theory/k_nearest_neighbor.html', {'data': json.dumps(json_data)})


def linear_regression(request):
    return render(request, 'algorithm/theory/linear_regression.html')


def decision_tree(request):
    if request.method == 'GET':

        data = load_iris()
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(data.data, data.target)

        def rules(clf, features, labels, node_index=0):
            node = {}
            if clf.tree_.children_left[node_index] == -1:  # indicates leaf
                count_labels = zip(clf.tree_.value[node_index, 0], labels)
                node['name'] = ', '.join(('{} of {}'.format(int(count), label) for count, label in count_labels))
            else:
                feature = features[clf.tree_.feature[node_index]]
                threshold = clf.tree_.threshold[node_index]
                node['name'] = '{} > {}'.format(feature, threshold)
                left_index = clf.tree_.children_left[node_index]
                right_index = clf.tree_.children_right[node_index]
                node['children'] = [rules(clf, features, labels, right_index),
                                    rules(clf, features, labels, left_index)]
            return node

        json_data = rules(clf, data.feature_names, data.target_names)
    return render(request, 'algorithm/theory/decision_tree.html', {'data': json.dumps(json_data)})


def deep_learnning(request):
    return render(request, 'algorithm/theory/deep_learnning.html')


def ensemble_learning(request):
    return render(request, 'algorithm/theory/ensemble_learning.html')


def naive_bayes(request):
    return render(request, 'algorithm/theory/naive_bayes.html')


def logistic_regression(request):
    return render(request, 'algorithm/theory/logistic_regression.html')


def support_vector_machine(request):
    return render(request, 'algorithm/theory/support_vector_machine.html')


def neural_network(request):
    return render(request, 'algorithm/theory/neural_network.html')


def k_means(request):
    return render(request, 'algorithm/theory/k_means.html')
