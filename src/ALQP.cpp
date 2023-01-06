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
    ceq = Eigen::VectorXd::Zero(A.rows());
    cinq = Eigen::VectorXd::Zero(C.rows());
    n = mu.rows();
    Irho = Eigen::MatrixXd::Zero(n,n);
    n_iters = 0;

}

double ALQP::AL(Eigen::VectorXd deltaX, double alpha)
{
    // Merit Function: Augmented Lagrangian
    Eigen::VectorXd xU = x + alpha*deltaX;
    Eigen::VectorXd ceqU = constraint_equality(xU);
    Eigen::VectorXd cinU = constraint_inequality(xU);
    
    return  0.5 * (xU).transpose() * P * (xU) + q.dot(xU) 
        + lambda.dot(ceqU) + mu.dot(cinU) + 0.5*rho*ceqU.transpose()*ceqU + 0.5*(Irho*cinU).transpose()*cinU;
}

void ALQP::active_constraints()
{
    ceq = constraint_equality(x);
    cinq = constraint_inequality(x);

    Irho.setZero();
    // Check for the active contraints
    for(int i=0; i<n; i++)
    {
        // if constraint is active or the dual variable associated with it is not 0
        if (cinq[i] > 0 || mu[i] != 0)
        {
            Irho(i,i) = rho;
        }
    }
}



// Computes the gradient and hessian of the Augmented Lagrangian
void ALQP::algrad()
{
    // Analytic gradient of AL
    g = (P * x + q + (A.transpose() * lambda) + (C.transpose() * mu)).transpose() + (rho * ceq).transpose() * A +
        (Irho * cinq).transpose() * C;
}

void ALQP::alhess()
{
    // Analytic hessian of AL
    H = P + (Irho * C).transpose() * C + (rho * A).transpose() * A;
}

Eigen::VectorXd ALQP::constraint_equality(Eigen::VectorXd xv)
{
    return A*xv - b;
}

Eigen::VectorXd ALQP::constraint_inequality(Eigen::VectorXd xv)
{
    return C*xv - d;
}

// Update State using Newton's Method
void ALQP::primal_update(float tol)
{
    for(int i=10; i--; )
    {
        // Get active constraint matrix
        active_constraints();

        // Compute gradient and hessian of AL
        algrad();
        alhess();

        // Check if the gradient is 0 (a.k.a minima)
        if (g.norm() < tol)
        {
            break;
        }

        Eigen::VectorXd deltaX = -H.inverse()*g;

        double alpha = 1.0;
        float alpha_scaling = 0.5;
        float beta = 0.1;

        while(AL(deltaX, alpha) > AL(deltaX, 0.0) + beta*alpha*g.dot(deltaX))
        {
            alpha *= alpha_scaling;
        }

        x += alpha*deltaX; 

        n_iters += 1;
    }
    

}

void ALQP::dual_update()
{
    lambda += rho*constraint_equality(x);
    mu += (rho*constraint_inequality(x)).cwiseMax(0);
}

void ALQP::solve(int max_iters)
{
    for(int i=max_iters; i--; )  //for(int i=0; i < max_iters; i++)
    {
        // Update primals (inner loop is unconstrained)
        primal_update();

        // update duals using updated primals (outer loop)
        dual_update();

        rho = phi*rho;
        
        if (constraint_equality(x).norm() < tol_main && (constraint_inequality(x).cwiseMax(0)).norm() < tol_main)
        {
            break;
        }
    }
}

Eigen::VectorXd ALQP::get_primal()
{
    return x;
}

double ALQP::get_cost()
{
    return 0.5 * (x).transpose() * P * (x) + q.dot(x);
} 