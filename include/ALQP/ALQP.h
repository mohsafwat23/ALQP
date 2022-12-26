#pragma once
#include <Eigen/Dense>

class ALQP
{
    public:
        
        Eigen::MatrixXd P;

        Eigen::VectorXd q;

        Eigen::MatrixXd A;

        Eigen::VectorXd b;

        Eigen::MatrixXd C;

        Eigen::VectorXd d;

        float rho;

        Eigen::VectorXd g;      // gradient of AL placeholder (might just make function Eigen:: ())

        Eigen::MatrixXd H;      // hessian of AL placeholder


        ALQP(Eigen::MatrixXd Pin, Eigen::VectorXd qin, Eigen::MatrixXd Ain, Eigen::VectorXd bin, Eigen::MatrixXd Cin, Eigen::VectorXd din, float rho = 1.0); // Constructor

        void solve();


    
    private:

        void algradhess(Eigen::VectorXd x, Eigen::VectorXd lambda, Eigen::VectorXd mu);
        
        void primal_update();

        void dual_update();

        Eigen::VectorXd constraint_equality(Eigen::VectorXd x);

        Eigen::VectorXd constraint_inequality(Eigen::VectorXd x);

        void linesearch();
};
