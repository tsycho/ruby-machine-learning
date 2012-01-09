require 'matrix'

module MachineLearning
  module MatrixFunctions
    def sigmoid(mZ)
      # 1 ./ (1 + exp(-z));
      mZ.map { |z| 1.0 / (1.0 + Math.exp(-z)) }
    end
    
    def one_by(mZ)
      mZ.map { |z| 1.0 / z }
    end
    
    def one_minus(mZ)
      mZ.map { |z| 1.0 - z }
    end

    def log(mZ)
      mZ.map { |z| Math.log(z) }
    end

    def matrix_size(mZ)
      "#{mZ.row_size}x#{mZ.column_size}"
    end
    
    # Alternative for ruby 1.8.7, since it does not have the Matrix.build method
    def build_matrix(m, n, &block)
      arr = Array.new(m) { Array.new(n) { 0.0 } }
      (0...m).each { |i|
        (0...n).each { |j|
          arr[i][j] = block.call(i,j)
        }
      }
      return Matrix.[](*arr)
    end    
  end
end