#include <iostream>
#include "ALQP/ALQP.h"

//void printhello() {}
ALQP::ALQP(Eigen::MatrixXd Pin, Eigen::VectorXd qin, Eigen::MatrixXd Ain, Eigen::VectorXd bin, Eigen::MatrixXd Cin, Eigen::VectorXd din, float rhoin)
{
    P = Pin;
    q = qin;
    A = Ain;
    b = bin;
    C = Cin;
    d = din;
    rho = rhoin;
}

// Computes the gradient and hessian of the Augmented Lagrangian
void ALQP::algradhess(Eigen::VectorXd x, Eigen::VectorXd lambda, Eigen::VectorXd mu)
{
    int n = mu.rows();
    Eigen::MatrixXd Irho(n,n);
    Eigen::VectorXd ceq = constraint_equality(x);
    Eigen::VectorXd cinq = constraint_inequality(x);

    // Check for the active contraints
    for(int i=0; i<n; i++)
    {
        // if constraint is active and the dual variable associated with it is not 0
        if (cinq[i] > 0 && mu[i] != 0)
        {
            Irho(i,i) = rho;
        }
    }
    // Analytic gradient
    g = P * x + q + A.transpose() * lambda + C.transpose() * mu + 
        (Irho * cinq).transpose() * C + (rho * ceq).transpose() * A;

    // Analytic hessian
    H = P + (Irho * C).transpose() * C + (rho * A).transpose() * A;
}

Eigen::VectorXd ALQP::constraint_equality(Eigen::VectorXd x)
{
    return A*x - b;
}

Eigen::VectorXd ALQP::constraint_inequality(Eigen::VectorXd x)
{
    return C*x - d;
}

// Update State using Newton's Method
void ALQP::primal_update(Eigen::VectorXd x, Eigen::VectorXd lambda, Eigen::VectorXd mu, float tol)
{
    for(int i=0; i<15; i++)
    {
        algradhess(x, lambda, mu);
        if (g.norm() < tol)
        {
            break;
        }
        x += -H.inverse()*g.transpose(); // CHECK THIS MIGHT NOT BE TRANSPOSED
    }
    

}

void ALQP::dual_update(Eigen::VectorXd x, Eigen::VectorXd lambda, Eigen::VectorXd mu)
{
    lambda += rho*constraint_equality(x);
    mu += (rho*constraint_equality(x)).cwiseMax(0);
}

