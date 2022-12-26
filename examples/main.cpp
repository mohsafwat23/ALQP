#include <iostream>
#include "ALQP/ALQP.h"
#include <chrono>

int main(){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    //Eigen::MatrixXd M = Eigen::MatrixXd::Random(3,3);
    Eigen::MatrixXd M(3,3);
    M << 2, 0.5, 8,
        3, 15, 2,
        2, 2, 4;

    Eigen::MatrixXd P = M.transpose() * M;

    //Eigen::VectorXd q = Eigen::VectorXd::Random(3);
    Eigen::VectorXd q(3);
    q << 1,2,3;

    //Eigen::MatrixXd A = Eigen::MatrixXd::Random(1,3);
    Eigen::MatrixXd A(1,3);
    A << 1,4,5;

    //Eigen::VectorXd b = Eigen::VectorXd::Random(1);
    Eigen::VectorXd b(1);
    b << 2;

    //Eigen::MatrixXd C = Eigen::MatrixXd::Random(3,3);
    Eigen::MatrixXd C(3,3);
    C << 1, 2, 4, 
        1, 5, 2, 
        8, 4, 3;

    //Eigen::VectorXd d = Eigen::VectorXd::Random(3);
    Eigen::VectorXd d(3);
    d << 2,5,7;

    Eigen::VectorXd x =  Eigen::VectorXd::Zero(3);

    ALQP opt(x,P,q,A,b,C,d);

    auto t0 = high_resolution_clock::now();
    opt.solve();
    auto t1 = high_resolution_clock::now();
    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t1 - t0);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t1 - t0;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

    std::cout << opt.x << "\n";

    return 0;
}