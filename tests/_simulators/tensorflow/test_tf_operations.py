#
# Copyright 2021-2025 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pytest

# This test will be skipped if TensorFlow is not available or not compatible
@pytest.mark.tensorflow
def test_basic_tensorflow_operations(tf):
    """Test basic TensorFlow operations using the tf fixture."""
    # Create a simple computation
    a = tf.constant(2.0)
    b = tf.constant(3.0)
    c = a * b
    
    assert c.numpy() == 6.0

# This test will be skipped if TensorFlow is not available or not compatible
@pytest.mark.tensorflow
def test_tensorflow_with_numpy(tf):
    """Test interoperability between TensorFlow and NumPy."""
    # Create a NumPy array
    np_array = np.array([1.0, 2.0, 3.0])
    
    # Convert to TensorFlow tensor
    tf_tensor = tf.convert_to_tensor(np_array)
    
    # Perform operations
    result = tf_tensor * 2
    
    # Convert back to NumPy and verify
    np.testing.assert_array_equal(result.numpy(), np.array([2.0, 4.0, 6.0]))

# This test will be skipped if TensorFlow is not available or not compatible
@pytest.mark.tensorflow
def test_tensorflow_gradient_calculation(tf):
    """Test automatic differentiation in TensorFlow."""
    # Define a simple function
    def f(x):
        return x ** 2
    
    # Create a variable to track
    x = tf.Variable(3.0)
    
    # Compute gradient
    with tf.GradientTape() as tape:
        y = f(x)
    
    dy_dx = tape.gradient(y, x)
    
    assert y.numpy() == 9.0
    assert dy_dx.numpy() == 6.0

# This test will be skipped if TensorFlow is not available or not compatible
@pytest.mark.tensorflow
def test_tensorflow_in_piquasso(generate_unitary_matrix, tf):
    """Test TensorFlow integration with Piquasso operations."""
    import piquasso as pq
    
    # Create a simple circuit with TensorFlow variable
    d = 2
    theta = tf.Variable(0.1)
    
    simulator = pq.PureFockSimulator(
        d=d, 
        config=pq.Config(cutoff=3), 
        connector=pq.TensorflowConnector()
    )
    
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()
        pq.Q(0) | pq.Dgate(theta)  # Using TensorFlow variable here
        
        # Apply a beam splitter with a fixed angle
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi/4, phi=0.0)
    
    # This will execute the circuit with the TensorFlow variable
    state = simulator.execute(program).state
    
    # Verify the state is a valid quantum state
    assert state is not None
    
    # You can add more specific assertions based on expected behavior
    density_matrix = state.dm()
    assert density_matrix is not None
    assert not np.isnan(density_matrix).any()
    assert not np.isinf(density_matrix).any()
