#include <iostream>
#include "ALQP/ALQP.h"
#include <chrono>
#include "OsqpEigen/OsqpEigen.h"

int main(){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // Change random Matrices https://stackoverflow.com/questions/21292881/matrixxfrandom-always-returning-same-matrices

    // min 0.5 x'P*x + q'x
    // st. A*x = b
    // C*x ≤ d

    Eigen::MatrixXd M = Eigen::MatrixXd::Random(3,3);
    // Eigen::MatrixXd M(3,3);
    // M << 2, 0.5, 8,
    //     3, 15, 2,
    //     2, 2, 4;

    Eigen::MatrixXd P = M.transpose() * M;

    //std::cout << "P: " << P << "\n";

    Eigen::SparseMatrix<double> hessian;

    hessian = P.sparseView();

    Eigen::VectorXd q = Eigen::VectorXd::Random(3);

    //std::cout << "q: " << q << "\n";
    // Eigen::VectorXd q(3);
    // q << 1,2,3;

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(1,3);

   // std::cout << "A: " << A << "\n";
    // Eigen::MatrixXd A(1,3);
    // A << 1,4,5;

    Eigen::VectorXd b = Eigen::VectorXd::Random(1);
    //std::cout << "b: " << b << "\n";
    // Eigen::VectorXd b(1);
    // b << 2;

    Eigen::MatrixXd C = Eigen::MatrixXd::Random(3,3);

    //std::cout << "C: " << C << "\n";
    // Eigen::MatrixXd C(3,3);
    // C << 0.1, 2, 4, 
    //     1, 5, 2, 
    //     8, 4, 30;

    Eigen::VectorXd d = Eigen::VectorXd::Random(3);
    
    //std::cout << "d: " << d << "\n";
    // Eigen::VectorXd d(3);
    // d << 2,5,7;

    Eigen::VectorXd x =  Eigen::VectorXd::Zero(3);

    // ================================================ OSQP ================================================


    // Concatenate the constraints vertically
    Eigen::SparseMatrix<double> linearMatrix;
    Eigen::MatrixXd ConcatenatedConstraints(A.rows()+ C.rows(), A.cols()); // <-- D(A.rows() + B.rows(), ...)
    ConcatenatedConstraints << A, C;
    linearMatrix = ConcatenatedConstraints.sparseView();


    // Concatenate equality and inequality constraints
    Eigen::MatrixXd upperBound(b.rows() + d.rows(), 1);
    upperBound << b, d;

    Eigen::VectorXd dlower(d.rows());
    dlower << -OsqpEigen::INFTY,-OsqpEigen::INFTY,-OsqpEigen::INFTY;


    Eigen::MatrixXd lowerBound(b.rows() + d.rows(), 1);
    lowerBound << b, dlower;

    // instantiate the solver
    OsqpEigen::Solver solver;

    // settings
    //solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);

    // OSQPSolution vector
    Eigen::VectorXd OSQPSolution;

    // set the initial data of the QP solver
    solver.data()->setNumberOfVariables(x.rows());
    solver.data()->setNumberOfConstraints(A.rows() + C.rows());
    solver.data()->setHessianMatrix(hessian);
    solver.data()->setGradient(q);
    solver.data()->setLinearConstraintsMatrix(linearMatrix);
    solver.data()->setLowerBound(lowerBound);
    solver.data()->setUpperBound(upperBound);

    // // instantiate the solver
    solver.initSolver();

    solver.solveProblem();

    OSQPSolution = solver.getSolution();

    std::cout << "OSQP Solution: " << OSQPSolution << "\n";

    // ================================================ ALQP ================================================

    ALQP opt(x,P,q,A,b,C,d);

    //auto t0 = high_resolution_clock::now();
    opt.solve();
    //auto t1 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    //duration<double, std::milli> ms_double = t1 - t0;

    //std::cout << ms_double.count() << "ms\n";

    std::cout << "ALQP Solution: " << opt.x << "\n";

    std::cout << "ALQP Optimal Objective: " << opt.get_cost() << "\n";

    std::cout << "ALQP iterations: " << opt.n_iters << "\n";

    return 0;
}