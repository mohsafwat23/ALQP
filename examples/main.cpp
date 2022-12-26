#include <iostream>
#include "ALQP/ALQP.h"

int main(){
    Eigen::MatrixXd P(2,2);

    Eigen::VectorXd q(10);

    Eigen::MatrixXd A(2,2);

    Eigen::VectorXd b(2);

    Eigen::MatrixXd C(2,2);

    Eigen::VectorXd d(2);

    ALQP opt(P,q,A,b,C,d);
    //opt.x = Eigen::VectorXd::Zero(5);
    std::cout << opt.q.rows();
    return 0;
}