import numpy as np
import heapq

class Hypothesis(object):
    
    def __init__(self, sequence, max_sequence_length, local_score=0, alpha=.6, eos_id=1, pad_id=0):
        
        self._sequence = sequence
        self.max_sequence_length = max_sequence_length
        self.local_score = local_score
        self.alpha = alpha
        self.eos_id = eos_id
        self.pad_id = pad_id
        
    def next(self, next_token, probability):
        """
        returns new hypothesis
        """
        
        if self.finished:
            raise Exception("Hypothesis is finished")

        sequence = self._sequence + [next_token]
        local_score = self.local_score + np.log(probability)
        
        return self.__class__(
            sequence=sequence,
            max_sequence_length=self.max_sequence_length,
            local_score=local_score,
            alpha=self.alpha,
            eos_id=self.eos_id
        )
    
    @property
    def sequence(self):
        
        #return np.array(self._sequence)


        sequence = np.pad(
            self._sequence,
            (0, self.max_sequence_length - len(self._sequence)),
            'constant',
            constant_values=self.pad_id
        )
        
        return np.reshape(
            sequence,
            [1, self.max_sequence_length]
        )


    @property
    def score(self):

        return self.local_score / (
            np.power(len(self._sequence) + 5, self.alpha) /
            np.power(5.0 + 1, self.alpha)
        )

    @property
    def finished(self):
        
        if len(self._sequence) < 1:
            return False
        else:
            return self.eos_id == self._sequence[-1]

class BeamSearch(object):
    
    def __init__(self, model_fn, num_hypotheses, beam_size, max_sequence_length, alpha=.6, eos_id=1, pad_id=0):
        
        self.model_fn = model_fn
        self.num_hypotheses = num_hypotheses
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.alpha = alpha
        self.eos_id = eos_id
        self.pad_id = pad_id
        self._best_finished_hypothesis = None
        self.hypotheses = None
        self.input_length = None

    def top(self, array, key, k):
        """
        returns indexes of top K elements in array
        @key is function to access comparable attribute of element
        """

        return heapq.nlargest(
            k,
            range(len(array)),
            key
        )

    def prune(self, hypotheses):
        
        top_indexes = self.top(
            hypotheses,
            key= lambda x: hypotheses[x].local_score,
            k=self.num_hypotheses
        )
        
        top_hypotheses = [hypotheses[i] for i in top_indexes]
        top_hypothesis = max(hypotheses, key=lambda x: x.local_score)
        
        top_hypotheses = [
            h for h in top_hypotheses
            if (top_hypothesis.local_score - h.local_score) <= self.beam_size
        ]
        
        if self._best_finished_hypothesis:
            top_hypotheses = [
                h for h in top_hypotheses
                if (top_hypothesis.score - self._best_finished_hypothesis.local_score) <= self.beam_size
            ]
        
        for h in top_hypotheses:
            if h.finished:
                if h.normalized_score > self._best_finished_hypothesis.normalized_score:
                    self._best_finished_hypothesis = h
        
        return [h for h in top_hypotheses if not h.finished]

    def reset_hypotheses(self):
        
        self.hypotheses = [
            Hypothesis(
                sequence=[],
                max_sequence_length=self.max_sequence_length,
                local_score=0,
                alpha=self.alpha,
                eos_id=self.eos_id,
                pad_id=self.pad_id
            )
        ]

    @property
    def search_done(self):
        """
        Returns whether to keep running search
        """
        
        if len(self.hypotheses) == 0:
            return True
        elif len(self.hypotheses[0]._sequence) >= min([
            self.input_length + 50,
            self.max_sequence_length
        ]):
            return True
        return False

    def search(self, embedding_input, initialize_graph=False):
        self.reset_hypotheses()
        self.input_length = embedding_input.shape[1]
        with tf.Session() as sess:
            if initialize_graph:
                sess.run(tf.global_variables_initializer())
            while not self.search_done:
                new_hypotheses = []
                for hypothesis in self.hypotheses:
                    output = self.model_fn(
                        sess=sess,
                        encoder_input=embedding_input,
                        decoder_input=hypothesis.sequence
                    )

                    next_predictions = output[0, len(hypothesis.sequence), :]
                    next_prediction_indexes = self.top(
                        next_predictions,
                        key=lambda x: next_predictions[x],
                        k=self.num_hypotheses
                    )
                    for i in next_prediction_indexes:
                        new_hypotheses.append(
                            hypothesis.next(
                                next_token=i,
                                probability=next_predictions[i]
                            )
                        )
                        
                    new_hypotheses = self.prune(new_hypotheses)
                    self.hypotheses = new_hypotheses

        if self._best_finished_hypothesis:
            best_hypothesis = self._best_finished_hypothesis
        else:
            best_hypothesis = self.hypotheses[0]
        for h in self.hypotheses:
            if h.score > best_hypothesis.score:
                best_hypothesis = h
                raise Exception()
    
        return best_hypothesis

