#include "ALQP/ALQP.h"
#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include "OsqpEigen/OsqpEigen.h"

TEST(ALQPSolver, RandomValues) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // min 0.5 x'P*x + q'x
    // st. A*x = b
    // C*x ≤ d

    // Change random Matrices https://stackoverflow.com/questions/21292881/matrixxfrandom-always-returning-same-matrices

    Eigen::SparseMatrix<double> hessian;

    Eigen::MatrixXd M = Eigen::MatrixXd::Random(3,3);

    Eigen::MatrixXd P = M.transpose() * M;

    hessian = P.sparseView();

    Eigen::VectorXd q = Eigen::VectorXd::Random(3);

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(1,3);

    Eigen::VectorXd b = Eigen::VectorXd::Random(1);

    Eigen::MatrixXd C = Eigen::MatrixXd::Random(3,3);

    Eigen::VectorXd d = Eigen::VectorXd::Random(3);

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

    std::cout << "Upper Bound " << upperBound << "\n";


    Eigen::VectorXd dlower(d.rows());
    dlower << -OsqpEigen::INFTY,-OsqpEigen::INFTY,-OsqpEigen::INFTY;


    Eigen::MatrixXd lowerBound(b.rows() + d.rows(), 1);
    lowerBound << b, dlower;

    std::cout << "lower Bound " << lowerBound << "\n";


    // instantiate the solver
    OsqpEigen::Solver solver;

    // settings
    //solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(false);
    solver.settings()->setAbsoluteTolerance(1e-6);

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

    auto t0 = high_resolution_clock::now();
    solver.solveProblem();
    auto t1 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t1 - t0;

    OSQPSolution = solver.getSolution();

    std::cout << "OSQP speed: " << ms_double.count() << "ms\n";

    std::cout << "OSQP Solution: " << OSQPSolution << "\n";

    std::cout << "x: " << x << "\n";


    // ================================================ ALQP ================================================

    ALQP opt(x,P,q,A,b,C,d);

    // TEMP
    float rho = 10.0;
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(C.rows());
    Eigen::MatrixXd Irho = Eigen::MatrixXd::Zero(mu.rows(),mu.rows());
    Eigen::VectorXd ceq = Eigen::VectorXd::Zero(A.rows());
    Eigen::VectorXd cinq = Eigen::VectorXd::Zero(C.rows());
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(A.rows());
    //
    Eigen::VectorXd g;
    t0 = high_resolution_clock::now();
    opt.solve();
    t1 = high_resolution_clock::now();

    // g = (P * x + q + (A.transpose() * lambda) + (C.transpose() * mu)).transpose() + (rho * ceq).transpose() * A +
    //     (Irho * cinq).transpose() * C;

    /* Getting number of milliseconds as a double. */
    ms_double = t1 - t0;

    std::cout << "ALQP speed: " << ms_double.count() << "ms\n";

    std::cout << "ALQP Solution: " << opt.x << "\n";

    std::cout << "ALQP Optimal Objective: " << opt.get_cost() << "\n";

    std::cout << "ALQP iterations: " << opt.n_iters << "\n";


    // Expect equality.
    EXPECT_EQ(roundf((opt.x[0] * 1000) / 1000),roundf((OSQPSolution[0] * 1000) / 1000));
    EXPECT_EQ(roundf((opt.x[1] * 1000) / 1000),roundf((OSQPSolution[1] * 1000) / 1000));
    EXPECT_EQ(roundf((opt.x[2] * 1000) / 1000),roundf((OSQPSolution[2] * 1000) / 1000));
}