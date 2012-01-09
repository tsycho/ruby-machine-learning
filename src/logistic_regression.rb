require 'matrix'
require './matrix_functions.rb'

# Conventions: 
# variables starting with m are matrices, v are column vectors
# m: number of data sets, n: number of features
# mX: m x n
# vTheta: n x 1
# vY: m x 1
module MachineLearning
  class LogisticRegression
    include MachineLearning::MatrixFunctions

    ALPHA_DEFAULT = 0.2
    ALPHA_SCALING_FACTOR = 2.0
    ALPHA_SCALING_ITERATIONS = 50
    LAMBDA_DEFAULT = 0.1
    NUM_ITER_DEFAULT = 300
    THRESHOLD_DEFAULT = 0.5

    attr_accessor :start_time
    
    def initialize
      @start_time = Time.now
    end
    
    # Returns [cost, gradient] 
    # => cost of current solution, and gradient of the cost w.r.t. theta
    def cost_function(mX, vY, vTheta, options={})
      m = vY.row_size.to_f
      n = vTheta.row_size.to_f
      h_x = sigmoid( mX * vTheta )  # m x 1      
      
      # cost = (-1/m) * h_x'*vY + log(1-h_x)'*(1-vY)
      cost = (-1.0/m) * ( h_x.t*vY + log(one_minus(h_x)).t * one_minus(vY) )[0,0]
      
      # Regularization (regLambda is the regularization parameter)
      reg_lambda = options[:reg_lambda] || LAMBDA_DEFAULT
      # cost += (lambda/(2*m)) * (theta(2:n,1)' * theta(2:n,1));
      cost += (reg_lambda/(2.0*m)) * theta_square(vTheta)
      
      # Partial derivatives for theta
      diff = h_x - vY; # m x 1
      # grad = (1/m) * (diff' * mX)'
      grad = (1.0/m) * (diff.t * mX).t; # n x 1
      grad = grad - (reg_lambda/m) * vTheta;

      return [cost, grad]
    end

    
    # Perform gradient descent for {num_iter} iterations with {alpha} rate of descent
    # Alpha is scaled down by a factor of 2 every
    def gradient_descent(mX, vY, vTheta, options={})
      alpha = options[:alpha] || ALPHA_DEFAULT
      numIter = options[:num_iter] || NUM_ITER_DEFAULT      

      (1..numIter).each do |iter|
        cost, grad = cost_function(mX, vY, vTheta, options)
        alpha /= ALPHA_SCALING_FACTOR if (iter % ALPHA_SCALING_ITERATIONS == 0)
        vTheta = vTheta - alpha * grad
        
        #break if (Time.now - @start_time) > 14.0
        #puts "#{iter}:\tCost: #{cost}"
      end
      
      return vTheta
    end

    def normalize(mX_training, mX_test)      
      n = mX_training.column_size
      maxx = []
      minx = []
      indices = []    
      (0...n).each { |i|
        col = mX_training.column(i)

        cmax = col.to_a.max
        cmin = col.to_a.min      

        if ((cmax - cmin) > 0.0)
          indices.push(i)
          maxx.push(cmax)
          minx.push(cmin)
        end
      }

      return [mX_training, mX_test].map { |mX|
        # Use this row if you are using ruby 1.8.7
        # build_matrix(mX.row_size, indices.size+1) { |r, i|
        Matrix.build(mX.row_size, indices.size+1) { |r, i|
          # Re-add the first column
          # This is a hack, and a better way to add a 1-vector to a matrix should be used.
          c = i-1
          (i == 0) ? 1.0 : (mX[r,indices[c]] - minx[c]) / (maxx[c] - minx[c])
        }
      }
    end

    # Returns an array [ predicted_Y, probability ]
    def solve(mX_training, vY_training, mX_test, options={})
      threshold = options[:threshold] || THRESHOLD_DEFAULT       

      # Normalize data
      normX_training, normX_test = normalize(mX_training, mX_test)
      
      initial_theta = Matrix.column_vector( Array.new(normX_training.column_size) { 0.5 } )
      theta = gradient_descent(normX_training, vY_training, initial_theta, options)

      # Predict on test set
      h_x = sigmoid( normX_test * theta )  # m x 1
      pY = h_x.map { |y| y >= threshold ? 1 : 0 }

      return [ pY.to_a, h_x.to_a, theta.to_a ]
    end
    
  private
    def theta_square(vTheta)
      n = vTheta.row_size
      # Note: vTheta[0,0] is not regularized
      return (1...n).inject(0) { |sum, i|
        sum += (vTheta[i,0] * vTheta[i,0])
      }
    end
    
  end
end