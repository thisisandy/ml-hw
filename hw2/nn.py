\usepackage{hyperref} % Add this line in the preamble to include the hyperref package

\subsection{Neural Network Optimization Problem}

The neural network (NN) optimization problem presents a unique challenge as a continuous problem, which inherently makes it less suitable for randomized optimization (RO) algorithms compared to discrete problems. Our analysis of four algorithms - Randomized Hill Climbing (RHC), Simulated Annealing (SA), Genetic Algorithm (GA), and Backpropagation - reveals distinct performance characteristics:

\textbf{RHC (Randomized Hill Climbing) and SA (Simulated Annealing):}
Figures \ref{fig:nn_restarts_tuning}, \ref{fig:nn_exp_const_tuning}, \ref{fig:nn_init_temp_tuning}, and \ref{fig:nn_min_temp_tuning} demonstrate that RHC and SA exhibit \uline{low sensitivity to hyperparameter tuning}. Their accuracy consistently remains below 0.9, significantly underperforming compared to GA and backpropagation. This suggests that these algorithms \uline{struggle to navigate the continuous solution space effectively} for neural network optimization.

\textbf{GA (Genetic Algorithm) and Backpropagation:} 
As evidenced in Figures \ref{fig:nn_mutation_prob_tuning} and \ref{fig:nn_pop_size_tuning}, both GA and backpropagation achieve \uline{high accuracy}, consistently exceeding 0.9. GA demonstrates the \uline{highest accuracy} among all tested algorithms. However, GA's superior performance comes at the cost of \uline{substantially increased computational time} compared to backpropagation and other algorithms.

\textbf{Performance and Efficiency Analysis:}
Figure \ref{fig:nn_results} provides a clear comparison of accuracy versus training time:
\begin{itemize}
\item GA achieves the \uline{highest accuracy} but at the expense of significantly longer training times.
\item Backpropagation offers a \uline{more balanced performance}, maintaining high accuracy with much shorter training times.
\item This indicates that backpropagation is the \uline{most efficient algorithm} for solving the NN problem, offering an optimal balance between accuracy and computational efficiency.
\end{itemize}

\textbf{Log Loss Performance:}
Figure \ref{fig:nn_fitness_curve} offers insights into log loss performance:
\begin{itemize}
\item GA \uline{outperforms backpropagation} in terms of log loss.
\item RHC and SA \uline{significantly underperform}, further confirming their ineffectiveness for this task.
\end{itemize}

\textbf{Cross-Validation Impact:}
Figure \ref{fig:nn_learning_curve} reveals interesting dynamics in cross-validation performance:
\begin{itemize}
\item RHC and SA \uline{quickly match the performance} of GA and backpropagation.
\item This unexpected behavior may be attributed to the \uline{small dataset size} (approximately 150 entries).
\item The limited data likely causes \uline{instability and performance degradation} in GA and backpropagation, which typically benefit from larger datasets.
\end{itemize}

In conclusion, while GA demonstrates superior accuracy in neural network optimization, backpropagation emerges as the \uline{most effective method} due to its balance of high accuracy and computational efficiency. The performance of RHC and SA suggests that these algorithms are not well-suited for continuous optimization problems like neural network weight optimization. The impact of dataset size on algorithm performance highlights the importance of considering data volume when selecting optimization methods for neural networks.

\section{Experiment Results}

This section presents the results of the experiments conducted using the three randomized optimization algorithms (RHC, SA, GA, and MIMIC) and their comparison with traditional backpropagation. The results are divided into two parts: optimization problems and neural network optimization.

\subsection{Optimization Problems Results}

\subsubsection{NQueen Problem}
The performance of the algorithms on the NQueen problem for different values of \(N\) is shown in Figure \ref{fig:nq_performance_vs_N}. The best fitness values achieved by each algorithm as \(N\) varies are plotted to illustrate how the problem size impacts the performance.

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nq_performance_vs_N.png}{\includegraphics[width=\linewidth]{images/nq_performance_vs_N.png}}
\caption{Performance of Algorithms as \(N\) Varies for the NQueen Problem}
\label{fig:nq_performance_vs_N}
\end{figure}

The execution times for each algorithm configuration are shown in Figure \ref{fig:nq_time_comparison}. Comparison of execution times highlights the computational efficiency of each algorithm. Please note that the y-axis uses a logarithmic scale, displaying $\log(\text{time})$ in milliseconds.

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nq_time_comparison.png}{\includegraphics[width=\linewidth]{images/nq_time_comparison.png}}
\caption{Execution Time Comparison for Different Configurations in the NQueen Problem}
\label{fig:nq_time_comparison}
\end{figure}

The fitness curves for each algorithm with different configurations are shown in Figure \ref{fig:nq}. These plots demonstrate how each algorithm's performance evolves over iterations.

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nq.png}{\includegraphics[width=\linewidth]{images/nq.png}}
\caption{Fitness Curves for Different Algorithm Configurations in the NQueen Problem}
\label{fig:nq}
\end{figure}

\subsubsection{One-Max Problem}
The performance of the algorithms on the One-Max problem for different values of \(N\) is shown in Figure \ref{fig:one_max_performance_vs_N}. The best fitness values achieved by each algorithm as \(N\) varies are plotted to illustrate how the problem size impacts the performance.

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/one_max_performance_vs_N.png}{\includegraphics[width=\linewidth]{images/one_max_performance_vs_N.png}}
\caption{Performance of Algorithms as \(N\) Varies for the One-Max Problem}
\label{fig:one_max_performance_vs_N}
\end{figure}

The execution times for each algorithm configuration are shown in Figure \ref{fig:one_max_times}. Comparison of execution times highlights the computational efficiency of each algorithm.

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/one_max_times.png}{\includegraphics[width=\linewidth]{images/one_max_times.png}}
\caption{Execution Time Comparison for Different Configurations in the One-Max Problem}
\label{fig:one_max_times}
\end{figure}

The fitness curves for each algorithm with different configurations are shown in Figure \ref{fig:one_max}. These plots demonstrate how each algorithm's performance evolves over iterations.

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/one_max.png}{\includegraphics[width=\linewidth]{images/one_max.png}}
\caption{Fitness Curves for Different Algorithm Configurations in the One-Max Problem}
\label{fig:one_max}
\end{figure}

\subsection{Neural Network Optimization Results}

The overall comparison of accuracy, F1 score, training time, and prediction time for the different algorithms is shown in Figure \ref{fig:nn_results}. These comparisons help to understand the efficiency and effectiveness of each algorithm in optimizing the neural network.

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nn_results.png}{\includegraphics[width=\linewidth]{images/nn_results.png}}
\caption{Comparison of Accuracy, F1 Score, Training Time, and Prediction Time for Different Algorithms in Neural Network Optimization}
\label{fig:nn_results}
\end{figure}

\subsubsection{Hyperparameter Tuning Results}
The impact of different hyperparameters on the neural network optimization performance is shown in Figures \ref{fig:nn_exp_const_tuning}, \ref{fig:nn_init_temp_tuning}, \ref{fig:nn_min_temp_tuning}, and \ref{fig:nn_restarts_tuning}. These plots illustrate how various hyperparameters affect the accuracy, F1 score, training time, and prediction time.

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nn_exp_const_tuning.png}{\includegraphics[width=\linewidth]{images/nn_exp_const_tuning.png}}
\caption{Impact of Different Exponential Constants on Neural Network Optimization}
\label{fig:nn_exp_const_tuning}
\end{figure}


\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nn_init_temp_tuning.png}{\includegraphics[width=\linewidth]{images/nn_init_temp_tuning.png}}
\caption{Impact of Different Initial Temperatures on Neural Network Optimization}
\label{fig:nn_init_temp_tuning}
\end{figure}

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nn_restarts_tuning.png}{\includegraphics[width=\linewidth]{images/nn_restarts_tuning.png}}
\caption{Impact of Different Restarts on Neural Network Optimization}
\label{fig:nn_restarts_tuning}
\end{figure}

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nn_min_temp_tuning.png}{\includegraphics[width=\linewidth]{images/nn_min_temp_tuning.png}}
\caption{Impact of Different Minimum Temperatures on Neural Network Optimization}
\label{fig:nn_min_temp_tuning}
\end{figure}

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nn_pop_size_tuning.png}{\includegraphics[width=\linewidth]{images/nn_pop_size_tuning.png}}
\caption{Impact of Different Population Sizes on Neural Network Optimization}
\label{fig:nn_pop_size_tuning}
\end{figure}

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nn_mutation_prob_tuning.png}{\includegraphics[width=\linewidth]{images/nn_mutation_prob_tuning.png}}
\caption{Impact of Different Mutation Probabilities on Neural Network Optimization}
\label{fig:nn_mutation_prob_tuning}
\end{figure}

\subsubsection{Fitness Curve Comparison}
The fitness curve comparison for neural network optimization using different algorithms is shown in Figure \ref{fig:nn_fitness_curve}. This plot compares how the fitness changes over iterations for each algorithm.

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nn_fitness_curve.png}{\includegraphics[width=\columnwidth]{images/nn_fitness_curve.png}}
\caption{Fitness Curve Comparison for Neural Network Optimization}
\label{fig:nn_fitness_curve}
\end{figure}

\subsubsection{Learning Curves}
The learning curves, showing training and test scores, fit times, and score times, for each algorithm are presented in Figure \ref{fig:nn_learning_curve}. These plots help visualize the learning behavior of the neural network under different optimization algorithms.

\begin{figure}[htbp]
\centering
\href{https://github.com/thisisandy/ml-hw/tree/hw2/hw2/nn_learning_curve.png}{\includegraphics[width=\columnwidth]{images/nn_learning_curve.png}}
\caption{Learning Curves for Training and Test Scores, Fit Times, and Score Times for Neural Network Optimization}
\label{fig:nn_learning_curve}
\end{figure}
