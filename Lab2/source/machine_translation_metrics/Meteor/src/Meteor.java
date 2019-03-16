/**
 * Wrapper for
 * Created by Gregory Brown 3/14/2019
 */
import edu.cmu.meteor.scorer.MeteorConfiguration;
import edu.cmu.meteor.scorer.MeteorScorer;
import edu.cmu.meteor.util.Constants;
import py4j.GatewayServer;

public class Meteor {

    MeteorScorer scorer;
    MeteorConfiguration config;

    public Meteor(){
        config = new MeteorConfiguration();
        config.setLanguage("en");
        config.setNormalization(Constants.NORMALIZE_KEEP_PUNCT);
        scorer = new MeteorScorer(config);
    }

    public double compute_score(String hyp, String ref){
        return scorer.getMeteorStats(hyp, ref).score;
    }

    public static void main(String[] args) {
        Meteor app = new Meteor();
//         app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(app);
        server.start();

        String ref_str = "the quick brown fox jumped over the lazy dog";
        String hyp_str = "the fast brown fox jumped over the lazy dog";
        System.out.println(app.compute_score(hyp_str, ref_str));
        System.out.println("test");
    }

}
