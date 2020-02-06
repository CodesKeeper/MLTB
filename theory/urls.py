from django.urls import path
from theory import views

app_name = 'theory'
urlpatterns = [
    path('least_square_method/', views.least_square_method, name='LSM'),
    path('batch_gradient_descent/', views.batch_gradient_descent, name='BGD'),
    path('stochastic_gradient_descent/', views.stochastic_gradient_descent, name='SGD'),
    path('linear_simulation_data/', views.linear_simulation_data, name='LSD'),
    path('classify_simulation_data/', views.classify_simulation_data, name='classify_data'),
    path('cluster_simulation_data/', views.cluster_simulation_data, name='cluster_data'),
    path('license_plate_recognition/', views.license_plate_recognition, name='LPR'),
    path('k_nearest_neighbor/', views.k_nearest_neighbor, name='KNN'),
    path('linear_regression/', views.linear_regression, name='LR'),
    path('decision_tree/', views.decision_tree, name='DT'),
    path('deep_learnning/', views.deep_learnning, name='DL'),
    path('naive_bayes/', views.naive_bayes, name='NB'),
    path('ensemble_learning/', views.ensemble_learning, name='EL'),
    path('logistic_regression/', views.logistic_regression, name='LOR'),
    path('support_vector_machine/', views.support_vector_machine, name='SVM'),
    path('neural_network/', views.neural_network, name='NN'),
    path('k_means/', views.k_means, name='KM'),
]
