import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def Rx(q):
    T = np.array([[1,         0,          0, 0],
                  [0, np.cos(q), -np.sin(q), 0],
                  [0, np.sin(q),  np.cos(q), 0],
                  [0,         0,          0, 1]], dtype=float)
    return T


def dRx(q):
    T = np.array([[0,          0,          0, 0],
                  [0, -np.sin(q), -np.cos(q), 0],
                  [0,  np.cos(q), -np.sin(q), 0],
                  [0,          0,          0, 0]], dtype=float)
    return T


def Ry(q):
    T = np.array([[ np.cos(q), 0, np.sin(q), 0],
                  [         0, 1,         0, 0],
                  [-np.sin(q), 0, np.cos(q), 0],
                  [         0, 0,         0, 1]], dtype=float)
    return T


def dRy(q):
    T = np.array([[-np.sin(q), 0,  np.cos(q), 0],
                  [         0, 0,          0, 0],
                  [-np.cos(q), 0, -np.sin(q), 0],
                  [         0, 0,          0, 0]], dtype=float)
    return T


def Rz(q):
    T = np.array([[np.cos(q), -np.sin(q), 0, 0],
                  [np.sin(q),  np.cos(q), 0, 0],
                  [        0,          0, 1, 0],
                  [        0,          0, 0, 1]], dtype=float)
    return T


def dRz(q):
    T = np.array([[-np.sin(q), -np.cos(q), 0, 0],
                  [ np.cos(q), -np.sin(q), 0, 0],
                  [         0,          0,  0, 0],
                  [         0,          0,  0, 0]], dtype=float)
    return T


def Tx(x):
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dTx(x):
    T = np.array([[0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=float)
    return T


def Ty(y):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, y],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dTy(y):
    T = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=float)
    return T


def Tz(z):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dTz(z):
    T = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]], dtype=float)
    return T

def fk(q, theta, link):
    H = np.linalg.multi_dot([Rz(q[0]), # active joint 
                             Rz(theta[0]), # 1 DOF virtual spring
                             Tz(link[0]), # rigid link
                             Tz(q[1]), # active joint 
                             Tz(theta[1]), # 1 DOF virtual spring
                             Tx(link[1]), # rigid link
                             Tx(q[2]), # active joint 
                             Tx(theta[2]) # 1 DOF virtual spring
                            ])
    return H

def ik(tool, link):
    x = tool[0]
    y = tool[1]
    z = tool[2]

    q1 = np.arctan2(y, x)
    q2 = z - link[0]
    q3 = np.sqrt(x**2 + y**2) - link[1]
    
    q = np.array([q1, q2, q3], dtype=float)
    return q

def jacobianTheta(q, theta, link):
    H = fk(q, theta, link)
    H[0:3, 3] = 0
    inv_H = np.transpose(H)

    dH = np.linalg.multi_dot([Rz(q[0]), # active joint 
                              dRz(theta[0]), # 1 DOF virtual spring
                              Tz(link[0]), # rigid link
                              Tz(q[1]), # active joint 
                              Tz(theta[1]), # 1 DOF virtual spring
                              Tx(link[1]), # rigid link
                              Tx(q[2]), # active joint 
                              Tx(theta[2]) # 1 DOF virtual spring
                             ])

    dH = np.linalg.multi_dot([dH, inv_H])
    J1 = np.vstack([dH[0, 3], dH[1, 3], dH[2, 3], dH[2, 1], dH[0, 2], dH[1, 0]])

    dH = np.linalg.multi_dot([Rz(q[0]), # active joint 
                              Rz(theta[0]), # 1 DOF virtual spring
                              Tz(link[0]), # rigid link
                              Tz(q[1]), # active joint 
                              dTz(theta[1]), # 1 DOF virtual spring
                              Tx(link[1]), # rigid link
                              Tx(q[2]), # active joint 
                              Tx(theta[2]) # 1 DOF virtual spring
                             ])

    dH = np.linalg.multi_dot([dH, inv_H])
    J2 = np.vstack([dH[0, 3], dH[1, 3], dH[2, 3], dH[2, 1], dH[0, 2], dH[1, 0]])

    dH = np.linalg.multi_dot([Rz(q[0]), # active joint 
                              Rz(theta[0]), # 1 DOF virtual spring
                              Tz(link[0]), # rigid link
                              Tz(q[1]), # active joint 
                              Tz(theta[1]), # 1 DOF virtual spring
                              Tx(link[1]), # rigid link
                              Tx(q[2]), # active joint 
                              dTx(theta[2]) # 1 DOF virtual spring
                             ])

    dH = np.linalg.multi_dot([dH, inv_H])
    J3 = np.vstack([dH[0, 3], dH[1, 3], dH[2, 3], dH[2, 1], dH[0, 2], dH[1, 0]])

    J = np.hstack([J1, J2, J3])
    return J


link = np.array([2, 1], dtype=float)
theta = np.array([0, 0, 0], dtype=float)

kTheta = np.array([1e+6, 2e+6, 0.5e+6], dtype=float)
kTheta = np.diag(kTheta)

#H = fk(q, theta, link)
#q = ik(H, link)


experiments = 30
first_term = np.zeros((3, 3), dtype=float)
second_term = np.zeros(3, dtype=float)

for i in range(experiments):
    q_revolute = np.random.uniform(-np.pi, np.pi, 1)
    q_prismatic = np.random.uniform(0, 1, 2)
    q = np.hstack([q_revolute, q_prismatic])
    W = np.random.uniform(-1000, 1000, 6)

    jTheta = jacobianTheta(q, theta, link)
    dt = np.linalg.multi_dot([jTheta, np.linalg.inv(kTheta), np.transpose(jTheta), W]) + np.random.normal(loc=0.0, scale=1e-5)

    jTheta = jTheta[0:3, :]
    dt = dt[0:3]
    W = W[0:3]

    A = np.zeros(jTheta.shape, dtype=float)
    for i in range(jTheta.shape[1]):
        j = jTheta[:, i]
        A[:, i] = np.outer(j, j).dot(W)

    first_term = first_term + np.transpose(A).dot(A)
    second_term = second_term + np.transpose(A).dot(dt)

ks = np.linalg.inv(first_term).dot(second_term)
stiffness = np.divide(1, ks)
kTheta = np.diag(stiffness) # new stiffness matrix
print(stiffness)

W = np.array([-440, -1370, -1635, 0, 0, 0], dtype=float)

r = 0.1
xc = 1.1
yc = 0
zc = 2.3

points = 50
alpha = np.linspace(0, 2*np.pi, points, dtype=float)
X = xc + r*np.cos(alpha)
Y = yc + r*np.sin(alpha)
Z = zc*np.ones(points, dtype=float)
trajectorDesired = np.stack([X, Y, Z])

jointStates = np.zeros((3, points), dtype=float)
for i in range(points):
    tool  = np.array([X[i], Y[i], Z[i]], dtype=float)
    jointStates[:, i] = ik(tool, link)

trajectoryUncalibrated = np.zeros(trajectorDesired.shape, dtype=float)
for i in range(points):
    jTheta = jacobianTheta(jointStates[:, i], theta, link)
    dt = np.linalg.multi_dot([jTheta, np.linalg.inv(kTheta), np.transpose(jTheta), W]) + np.random.normal(loc=0.0, scale=1e-5)
    trajectoryUncalibrated[:, i] = trajectorDesired[:, i] + dt[0:3]

difference = trajectorDesired - trajectoryUncalibrated
trajectoryUpdated = trajectorDesired + difference

for i in range(points):
    tool  = np.array([trajectoryUpdated[0, i], trajectoryUpdated[1, i], trajectoryUpdated[2, i]], dtype=float)
    jointStates[:, i] = ik(tool, link)

trajectorCalibrated = np.zeros(trajectorDesired.shape, dtype=float)
for i in range(points):
    jTheta = jacobianTheta(jointStates[:, i], theta, link)
    dt = np.linalg.multi_dot([jTheta, np.linalg.inv(kTheta), np.transpose(jTheta), W]) + np.random.normal(loc=0.0, scale=1e-5)
    trajectorCalibrated[:, i] = trajectoryUpdated[:, i] + dt[0:3]

fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.plot3D(trajectorDesired[0], trajectorDesired[1], trajectorDesired[2], c='green', linewidth=2)
ax.scatter3D(trajectoryUncalibrated[0], trajectoryUncalibrated[1], trajectoryUncalibrated[2], c='blue', s=10)
ax.scatter3D(trajectorCalibrated[0], trajectorCalibrated[1], trajectorCalibrated[2], c='red', s=10)
trajectorDesiredPatch = mpatches.Patch(color='green', label='Desired trajectory')
trajectoryUncalibratedPatch = mpatches.Patch(color='blue', label='Uncalibrated trajectory')
trajectorCalibratedPatch = mpatches.Patch(color='red', label='Calibrated trajectory')
plt.legend(handles=[trajectorDesiredPatch, trajectoryUncalibratedPatch, trajectorCalibratedPatch])
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
plt.savefig('calibration.jpg', format="jpg")
plt.savefig('calibration.eps', format="eps")
plt.show()