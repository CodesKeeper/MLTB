from django.urls import path
from performance import views

app_name = 'performance'
urlpatterns = [
    path('decision_tree/', views.decision_tree, name='DT'),
    path('deep_learning/', views.deep_learning, name='DL'),
    path('ensemble_learning/', views.ensemble_learning, name='EL'),
    path('k_means/', views.k_means, name='KM'),
    path('k_nearest_neighbor/', views.k_nearest_neighbor, name='KNN'),
    path('linear_regression/', views.linear_regression, name='LR'),
    path('logistic_regression/', views.logistic_regression, name='LOR'),
    path('naive_bayes/', views.naive_bayes, name='NB'),
    path('neural_network/', views.neural_network, name='NN'),
    path('support_vector_machine/', views.support_vector_machine, name='SVM'),
]
