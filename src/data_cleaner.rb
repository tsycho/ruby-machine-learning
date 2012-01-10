require 'matrix'
require './matrix_functions'

module MachineLearning
  module DataCleaner

  	# Normalize both training and test data, according to training data
  	# Normalization is done according to max and min values for each column/feature
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

      normalized = [mX_training, mX_test].map { |mX|        
        # build_matrix(mX.row_size, indices.size+1) { |r, i|	# Use this row if you are using ruby 1.8.7
        Matrix.build(mX.row_size, indices.size) { |r, i|
          (mX[r,indices[c]] - minx[c]) / (maxx[c] - minx[c])
        }
      }

      return normalized.push(indices)
    end

    # Add a column of ones in front of the given matrices
    def add_one_vector(*matrices)
      matrices.map { |m|
      	num_rows = m.row_size
      	num_cols = m.column_size      	
      	columns = [ Array.new(num_rows) { 1.0 } ]
      	columns.push *(m.column_vectors.map { |v| v.to_a })
      	Matrix.columns(columns)
      }
    end

  end
end	