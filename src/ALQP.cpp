#include <iostream>
#include "ALQP/ALQP.h"

ALQP::ALQP(Eigen::VectorXd x0, Eigen::MatrixXd Pin, Eigen::VectorXd qin, Eigen::MatrixXd Ain, Eigen::VectorXd bin, Eigen::MatrixXd Cin, Eigen::VectorXd din, float rhoin)
{
    x = x0;
    P = Pin;
    q = qin;
    A = Ain;
    b = bin;
    C = Cin;
    d = din;
    rho = rhoin;
    phi = 10.0;
    tol_main = 1e-6;
    lambda = Eigen::VectorXd::Zero(A.rows());
    mu = Eigen::VectorXd::Zero(C.rows());
    n = mu.rows();

}

// Computes the gradient and hessian of the Augmented Lagrangian
void ALQP::algradhess()
{
    // int n = mu.rows();
    Eigen::MatrixXd Irho(n,n);
    Irho.setZero();
    Eigen::VectorXd ceq = constraint_equality();
    Eigen::VectorXd cinq = constraint_inequality();


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
    g = (P * x + q + A.transpose() * lambda + C.transpose() * mu).transpose() + (rho * ceq).transpose() * A +
        (Irho * cinq).transpose() * C;

    // Analytic hessian
    H = P + (Irho * C).transpose() * C + (rho * A).transpose() * A;
}

Eigen::VectorXd ALQP::constraint_equality()
{
    return A*x - b;
}

Eigen::VectorXd ALQP::constraint_inequality()
{
    return C*x - d;
}

// Update State using Newton's Method
void ALQP::primal_update(float tol)
{
    for(int i=0; i<10; i++)
    {
        // Compute gradient and hessian of AL
        algradhess();

        // Check if the gradient is 0 (a.k.a minima)
        if (g.norm() < tol)
        {
            break;
        }
        
        x += -H.inverse()*g; 

    }
    

}

void ALQP::dual_update()
{
    lambda += rho*constraint_equality();
    mu += (rho*constraint_inequality()).cwiseMax(0);
}

void ALQP::solve(int max_iters)
{
    for(int i=0; i < max_iters; i++)
    {
        // Update primals (inner loop is unconstrained)
        primal_update();

        // update duals using updated primals (outer loop)
        dual_update();

        rho = phi*rho;


        if (constraint_equality().norm() < tol_main && (constraint_inequality().cwiseMax(0)).norm() < tol_main)
        {
            break;
        }
    }
}

Eigen::VectorXd ALQP::get_primal()
{
    return x;
}
