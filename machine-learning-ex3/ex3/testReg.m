theta = [-2; -1; 1; 2];
theta.^2
thetaSq = eye(size(theta,1))*theta.^2
thetaSq(2:size(theta,1))

sum(thetaSq(2:size(theta,1)))



theta(2:size(theta,1))'*theta(2:size(theta,1))
theta'*theta-theta(1)^2