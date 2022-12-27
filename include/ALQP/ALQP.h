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

        Eigen::VectorXd x;

        Eigen::VectorXd lambda;

        Eigen::VectorXd mu;

        float rho;

        float phi;

        float tol_main;

        Eigen::VectorXd g;      // gradient of AL placeholder (might just make function Eigen:: ())

        Eigen::MatrixXd H;      // hessian of AL placeholder

        int n;

        ALQP(Eigen::VectorXd x0, Eigen::MatrixXd Pin, Eigen::VectorXd qin, Eigen::MatrixXd Ain, Eigen::VectorXd bin, Eigen::MatrixXd Cin, Eigen::VectorXd din, float rho = 10.0); // Constructor

        void solve(int max_iters = 20);

        Eigen::VectorXd get_primal();

        Eigen::MatrixXd Irho;

        Eigen::VectorXd ceq;

        Eigen::VectorXd cinq;
    
    private:

        void algrad();

        void alhess();
        
        void primal_update(float tol=1e-6);

        void dual_update();

        void active_constraints();

        Eigen::VectorXd constraint_equality(Eigen::VectorXd xv);

        Eigen::VectorXd constraint_inequality(Eigen::VectorXd xv);

        double AL(Eigen::VectorXd deltaX, double alpha);
};
