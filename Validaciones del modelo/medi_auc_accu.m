function [AUC,Accu,Sens,Spec,cm]=medi_auc_accu(Predict,Yt)
% Function for compute: AUC, Accuracy, sensitivity, specificity 
% Input: 
%        Prediction - predictive  label
%        Yt         - real label
% Output:
%        AUC:   Area under curve %Balanced Accuracy
%        Accu:  Accuracy
%        Sens:  Sensitivity
%        Spec:  Specifity
%        cm:  Confusion Matrix
Accu=1-(sum(Predict(:)~=Yt)/length(Predict(:)));
tPos=sum(Yt==1 & Predict(:)==1);
tNeg=sum(Yt==-1 & Predict(:)==-1);
fPos=sum(Yt==-1 & Predict(:)==1);
fNeg=sum(Yt==1 & Predict(:)==-1);
Sens=tPos/(tPos+fNeg);
Spec=tNeg/(fPos+tNeg);
AUC=(Sens+Spec)/2; %Balanced Accuracy
cm=[tNeg, fPos; fNeg, tPos];
end