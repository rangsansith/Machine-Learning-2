package hw6.svm.smo;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

public class TwoClassAllTesting {

	private static double tol2 = 0.01;
	private static double [] E;
	static int len=4601;
	static int featlen=57;
	static int classno=2;
	static int trainlen,testlen;
	static double[][] features = new double[len][featlen];
	static double[] labels = new double[len];
	static ArrayList<double[][]> featklist = new ArrayList<double[][]>();
	static ArrayList<double[]> labelsklist = new ArrayList<double[]>();
	static ArrayList<Integer> indexlist = new ArrayList<Integer>();
	static double[][] featurestrain = new double[1][featlen];
	static double [] labelstrain = new double[1];
	static double [][] featurestest= new double[1][featlen];
	static double [][] kernelmat= new double[1][1];
	static double [] labelstest= new double[1];
	static double[] alpha;
	static HashMap<Integer, Boolean> nonbound = new HashMap<Integer, Boolean>();
	static Set<Integer> nonboundlist = new HashSet<Integer>();
	static double b;
	static double regc=100;
	static double numtol=0.01;
	static int nonboundcount;
	private static void readData()
	{
		int one=0,zero=0;
		try {
			FileReader featureread = new FileReader("data/spambase.data");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				indexlist.add(ind);
				feats=sCurrentLine.split(",");
				for(int i=0;i<feats.length-1;i++)
				{
					features[ind][i]=Double.parseDouble(feats[i]);

				}
				labels[ind]=Double.parseDouble(feats[feats.length-1]);
				if(labels[ind]==0.0)
					zero++;
				else
					one++;
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		//System.out.println(zero+"   "+one+"  "+(double) zero/(zero+one));
	}
	private static void randSplits(int folds)
	{
		HashMap<Integer, Boolean> seen = new HashMap<Integer, Boolean>();
		int kfold=len/folds,listno=0,one=0,zero=0;
		Random n = new Random(); 
		while(indexlist.size()!=0){
			one=0;zero=0;
			int r=Math.min(kfold, indexlist.size());
			double[][] featurestemp = new double[r][featlen];
			double [] labelstemp = new double[r];
			for (int i = 0; i < kfold; i++) {
				if(indexlist.size()==0)
					break;
				int ind=n.nextInt(indexlist.size());
				int tmpind=indexlist.get(ind);
				if(seen.containsKey(tmpind))
					System.out.println("Error: bad Coder");
				else
					seen.put(tmpind, true);
				for (int j = 0; j < featlen; j++) {
					featurestemp[i][j]=features[tmpind][j];
				}
				labelstemp[i]=labels[tmpind];
				if(labels[tmpind]==0.0)
					zero++;
				else if(labels[tmpind]==1.0)
					one++;
				else
					System.out.println("Error: bad Coder");

				indexlist.remove(ind);
			}
			//System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
	}
	private static void uniformSplits(int folds){
		ArrayList<ArrayList<double[]>> tft= new ArrayList<ArrayList<double[]>>();
		ArrayList<ArrayList<Double>> tlt= new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < folds; i++) {
			tft.add(new ArrayList<double[]>());
			tlt.add(new ArrayList<Double>());
		}
		for (int i = 0; i < len; i++) {
			ArrayList<double[]> temp = tft.get(i%folds);
			temp.add(features[i]);
			tft.set(i%folds, temp);
			ArrayList<Double> lemp = tlt.get(i%folds);
			lemp.add(labels[i]);
			tlt.set(i%folds,lemp);
		}
		if(tft.size()!=tlt.size())
			System.out.println("Error: BadCode");
		int one=0,zero=0,listno=0;
		for (int i = 0; i < tft.size(); i++) {
			one=0;zero=0;
			double[][] featurestemp = new double[tft.get(i).size()][featlen];
			double [] labelstemp = new double[tlt.get(i).size()];
			for (int j = 0; j < tft.get(i).size(); j++) {
				featurestemp[j]=tft.get(i).get(j);
				labelstemp[j]=tlt.get(i).get(j);
				if(labelstemp[j]==0)
					zero++;
				else
					one++;
			}
			//System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
	}
	private static void pickKfold(int k){
		k--;
		testlen=labelsklist.get(k).length;
		trainlen=len-testlen;
		featurestrain=new double[trainlen][featlen];
		labelstrain=new double[trainlen];
		featurestest=new double[testlen][featlen];
		labelstest=new double[testlen];
		int ind=0;
		for (int i = 0; i < featklist.size(); i++) {
			if(i==k)
			{
				featurestest=featklist.get(k);
				labelstest=labelsklist.get(k);
				continue;
			}
			double[][] tf = featklist.get(i);
			double[] tl = labelsklist.get(i);
			for (int j = 0; j < tl.length; j++) {
				featurestrain[ind]=tf[j];
				labelstrain[ind]=tl[j];
				ind++;
			}
		}
	}
	private static double[] changeleabels(double[] labelsc){
		for (int i = 0; i < labelsc.length; i++) {
			if(labelsc[i]==0)
				labelsc[i]=-1;
		}
		return labelsc;
	}
	private static double[] LH(int i,int j){
		double[] ans={0,0};
		if(labelstrain[i]==labelstrain[j])
		{
			ans[0]=Math.max(0, alpha[j]+alpha[i]-regc);
			ans[1]=Math.min(regc,alpha[j]+alpha[i]);
		}
		else
		{
			ans[0]=Math.max(0, alpha[j]-alpha[i]);
			ans[1]=Math.min(regc, regc+alpha[j]-alpha[i]);
		}
		return ans;
	}
	private static void updatenonbound(int i,double val){
		if(val<regc&&val>0)
		{
			nonbound.put(i, true);
			nonboundlist.add(i);
		}
		else
		{
			    nonbound.put(i, false);
				nonboundlist.remove((Integer)i);
		}
	}

	private static int update(int i, int j) {
		double ai, aj;
		if (i == j) return 0;
		double bb = b;
		double alphai = alpha[i];
		double alphaj = alpha[j];
		int yi = (int) labelstrain[i];
		int yj = (int) labelstrain[j];
		double ei = E[i];
		double ej = E[j];
		double s = yi*yj;
		double[] LH = LH(i, j);
		double L = LH[0];
		double H = LH[1];
		if (L == H)
			return 0;
		double kii = kernelmat[i][i];
		double kjj = kernelmat[j][j];
		double kij = kernelmat[i][j]; 
		double eta = 2*kij-kii-kjj;
		if (eta < 0) {
			aj = alphaj - yj*(ei-ej)/eta;
			if (aj < L)
				aj = L;
			else if (aj > H)
				aj = H;
		} else {
			double vi, vj;
			vi = mpredict(i) - yi*alphai*kii - yj*alphaj*kij;
			vj = mpredict(j) - yi*alphai*kij - yj*alphaj*kjj;
			double Lobj = smoObj(alphai,alphaj,L, yi, yj, kij, kii, kjj, vi, vj);
			double Hobj = smoObj(alphai,alphaj,H, yi, yj, kij, kii, kjj, vi, vj);
			if (Lobj > Hobj+tol2)
				aj = L;
			else if (Lobj < Hobj-tol2)
				aj = H;
			else
				aj = alphaj;
		}
		if (Math.abs(aj-alphaj) < tol2*(aj+alphaj+tol2))
			return 0;
		ai = alphai + s*(alphaj-aj);
		double b1=b-ei-labelstrain[i]*(alpha[i]-alphai)*kernelmat[i][i]-labelstrain[j]*(alpha[j]-alphaj)*kernelmat[i][j];
		double b2=b-ej-labelstrain[i]*(alpha[i]-alphai)*kernelmat[i][j]-labelstrain[j]*(alpha[j]-alphaj)*kernelmat[j][j];
		if(alpha[i]<regc)
			b=b1;
		else if(alpha[j]<regc)
			b=b2;
		else
			b=(b1+b2)/2;
		alpha[i] = ai;
		alpha[j] = aj;
		//updatenonbound(j,alpha[j]);
		//updatenonbound(i,alpha[i]);
		for (int k=0; k<trainlen; k++) {
			double kik = kernelmat[i][k];  
			double kjk = kernelmat[j][k];  
			E[k] += (ai-alphai)*yi*kik + (aj-alphaj)*yj*kjk - bb + b;
		}
		return 1;
	}
	private static int newheur(int j) {
		int i = 0;
		int yj = (int) labelstrain[j];
		double alphaj = alpha[j];
		double ej = E[j];
		double rj = ej*yj;
		int randpos=0;
		nonboundcount=nonboundlist.size();
		if ((rj<-numtol && alphaj<regc) || (rj>numtol && alphaj>0)) {
			if(nonboundcount>1)
			{
				int maxind = 0;
				double maxval = Math.abs(E[0]-ej);
				for (int k=1; k<trainlen; k++)
					if (Math.abs(E[k]-ej) > maxval) {
						maxval = Math.abs(E[k]-ej);
						maxind = k;
					}
				if (update(maxind, j)==1)
					return 1;
			}
			randpos=(int)Math.floor(Math.random()*nonboundcount);
			for (int k = 0; k < nonboundcount; k++) {
				i = (randpos+k)%nonboundcount;
				//i=nonboundlist.get(i);	 
				if (update(i, j) == 1)
					return 1;
			}
			randpos = (int)Math.floor(Math.random()*trainlen);
			for (int k=0; k<alpha.length; k++) {
				i = (randpos+k)%trainlen;
				if (update(i, j) == 1)
					return 1;
			}
		}
		return 0;
	}
	private static int ExamineExample(int j) {
		int i = 0;
		int randpos;
		int yj = (int) labelstrain[j];
		double alphaj = alpha[j];
		double ej = E[j];
		double rj = ej*yj;
		if ((rj<-numtol && alphaj<regc) || (rj>numtol && alphaj>0)) {
			boolean exists = false;
			for (int k=0; k<trainlen; k++)
				if (alpha[k]>0 && alpha[k]<regc) {
					exists = true;
					break;
				}
			if (exists) {
				//second choice heuristics:
				int maxind = 0;
				double maxval = Math.abs(E[0]-ej);
				for (int k=1; k<trainlen; k++)
					if (Math.abs(E[k]-ej) > maxval) {
						maxval = Math.abs(E[k]-ej);
						maxind = k;
					}
				if (update(maxind, j) == 1)
					return 1;
			}
			//loop over non-zero & non-C alpha, starting at a random point:
			randpos = (int)Math.floor(Math.random()*trainlen);
			for (int k=0; k<alpha.length; k++) {
				i = (randpos+k)%trainlen;
				if (alpha[i]>0 && alpha[i]<regc) {
					if (update(i, j) == 1)
						return 1;
				}
			}
			//loop over all i, starting at a random point
			randpos = (int)Math.floor(Math.random()*trainlen);
			for (int k=0; k<alpha.length; k++) {
				i = (randpos+k)%trainlen;
				if (update(i, j) == 1)
					return 1;
			}
		}
		return 0;
	}
	private static int anotherheur(int j) {
		int i = 0;
		int randpos;
		int yj = (int) labelstrain[j];
		double alphaj = alpha[j];
		double ej = E[j];
		double rj = ej*yj;
		if ((rj<-numtol && alphaj<regc) || (rj>numtol && alphaj>0)) {
			boolean exists = false;
			for (int k=0; k<trainlen; k++)
				if (alpha[k]>0 && alpha[k]<regc) {
					exists = true;
					break;
				}
			if (exists) {
				//second choice heuristics:
				int maxind = 0;
				double maxval = Math.abs(E[0]-ej);
				for (int k=1; k<trainlen; k++)
					if (Math.abs(E[k]-ej) > maxval) {
						maxval = Math.abs(E[k]-ej);
						maxind = k;
					}
				if (update(maxind, j) == 1)
					return 1;
			}
			//loop over non-zero & non-C alpha, starting at a random point:
			randpos =0;// (int)Math.floor(Math.random()*trainlen);
			for (int k=0; k<alpha.length; k++) {
				i = k;
				if (alpha[i]>0 && alpha[i]<regc) {
					if (update(i, j) == 1)
						return 1;
				}
			}
			//loop over all i, starting at a random point
			randpos = 0;//(int)Math.floor(Math.random()*trainlen);
			for (int k=0; k<alpha.length; k++) {
				i = k;
				if (update(i, j) == 1)
					return 1;
			}
		}
		return 0;
	}
	private static int modheur(int j) {
		int i = 0;
		int yj = (int) labelstrain[j];
		double alphaj = alpha[j];
		double ej = E[j];
		double rj = ej*yj;
		int randpos=0;
		nonboundcount=nonboundlist.size();
		if ((rj<-numtol && alphaj<regc) || (rj>numtol && alphaj>0)) {
			if(nonboundcount>1)
			{
				int maxind = 0;
				double maxval = Math.abs(E[0]-ej);
				for (int k=1; k<trainlen; k++)
					if (Math.abs(E[k]-ej) > maxval) {
						maxval = Math.abs(E[k]-ej);
						maxind = k;
					}
				if (update(maxind, j)==1)
					return 1;
			}
			//randpos=(int)Math.floor(Math.random()*nonboundcount);
			Iterator<Integer> iter = nonboundlist.iterator();
			while(iter.hasNext())
			{
				i=iter.next();
				if (update(i, j) == 1)
					return 1;
			}
			//randpos = (int)Math.floor(Math.random()*trainlen);
			for (int k=0; k<alpha.length; k++) {
				i = k;
				if (update(i, j) == 1)
					return 1;
			}
		}
		return 0;
	}
	
	private static void SMO_Platt() {
		int numChanged = 0;
		int examineAll = 1;
		E = new double [trainlen];
		setcache();
		int runs=0;
		while (numChanged > 0 || examineAll == 1) {
			runs++;
			numChanged = 0;
			if (examineAll == 1) {
				for (int i=0; i<trainlen; i++) {
					//numChanged += ExamineExample(i);
					numChanged += anotherheur(i);
					//numChanged += modheur(i);
					//numChanged += newheur(i);//ExamineExample(i);
				}
			} else {
				for (int i=0; i<trainlen; i++) {
					if (alpha[i]!= 0 && alpha[i] != regc) {
						  	//numChanged += ExamineExample(i);
						  	numChanged += anotherheur(i);
						  	//numChanged += modheur(i);
						//numChanged += newheur(i);//ExamineExample(i);
					}
				}
			}
			if (examineAll == 1)
				examineAll = 0;
			else if (numChanged == 0)
				{
				examineAll = 1;
				double acc=0.0;
				for (int i = 0; i < trainlen; i++) {
					//System.out.println(mpredict(i)+" "+labelstrain[i]);
					if(mpredict(i)*labelstrain[i]>0)
						acc++;

				}
				System.out.println(acc/(double) trainlen);
				}
			System.out.println(runs+" "+numChanged);
			if(runs>1500)
				break;
		}
		System.out.println();
	}
	private static void setcache(){
		double error=0;
		for (int i = 0; i < trainlen; i++) {
			error=mpredict(i)-labelstrain[i];
			E[i]=error;
		}
	}

	private static double smoObj(double alphai,double alphaj,double aj, int yi, int yj, double kij, double kii, double kjj, double vi, double vj) {
		double s = yi*yj;
		double gamma = alphai + s*alphaj;
		return (gamma + (1-s)*aj - 0.5*kii*(gamma-s*aj)*(gamma-s*aj) - 0.5*kjj*aj*aj +
				- s*kij*(gamma-s*aj)*aj - yi*(gamma-s*aj)*vi - yj*aj*vj);
	}

	private static double mpredict(int xno){
		double ans=b;
		for (int i = 0; i < trainlen; i++) {
			ans+=alpha[i]*kernelmat[xno][i]*labelstrain[i];
		}
		return ans;
	}
	private static void writeforlib(){

		try {
			FileWriter fw = new FileWriter("resulthw6/ptrainlib.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i = 0; i < trainlen; i++) {
				bw.write(((int)labelstrain[i])+"");
				for (int j = 0; j < featlen; j++) {
					bw.write(" "+(j+1)+":"+featurestrain[i][j]);
				}
				bw.newLine();
			}
			bw.close();
			fw.close();

		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
		try {
			FileWriter fw = new FileWriter("resulthw6/ptestlib.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i = 0; i < testlen; i++) {
				bw.write(((int)labelstest[i])+"");
				for (int j = 0; j < featlen; j++) {
					bw.write(" "+(j+1)+":"+featurestest[i][j]);
				}
				bw.newLine();
			}
			bw.close();
			fw.close();

		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
	}
	
	private static void initalphas(){
		for (int i = 0; i < trainlen; i++) {
			alpha[i]=0.0;
		}
	}
	private static void kernel(){
		for(int i=0;i<trainlen;i++)
		{
			for(int j=0;j<trainlen;j++)
			{
				double ans=0;
				for(int k=0;k<featlen;k++)
				{
					ans+=featurestrain[i][k]*featurestrain[j][k];
				}
				kernelmat[i][j]=ans;
			}
		}
	}
	public static void main(String[] args) {
		trainlen=7;
		featlen=2;
		testlen=0;
		double f[][]={{0,2},{1,1},{2,2},{2,0},{0,0},{1,0},{0,1}};
		double t[]={1,1,1,1,0,0,0};
		featurestrain=f;
		labelstrain=t;
		writeforlib();
		labelstrain=changeleabels(labelstrain);			
		alpha=new double[trainlen];
		initalphas();
		kernelmat=new double[trainlen][trainlen];
		kernel();
		b=0;
		SMO_Platt();
		double acc=0.0;
		for (int i = 0; i < trainlen; i++) {
			System.out.println(mpredict(i)+" "+labelstrain[i]+" "+alpha[i]);
			if(mpredict(i)*labelstrain[i]>0)
				acc++;

		}
		System.out.println(acc/(double) trainlen);
		}
}
