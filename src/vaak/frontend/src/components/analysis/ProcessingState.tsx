import { Icon } from '../ui/Icon';
import { ListeningMark } from '../ui/ListeningMark';

interface ProcessingStateProps {
  processingStep: number;
}

export function ProcessingState({ processingStep }: ProcessingStateProps) {
  const steps = ["Validate", "Normalize", "Examine", "Aggregate"];
  const copyTitles = [
    "Receiving audio",
    "Preparing signal",
    "Evaluating temporal evidence",
    "Aggregating result",
    "Analysis complete"
  ];

  return (
    <div className="processing-state">
      <ListeningMark compact />
      <div className="processing-copy">
        <p>Vaak is listening across the recording</p>
        <h3>{copyTitles[processingStep]}</h3>
      </div>
      <ol>
        {steps.map((step, index) => (
          <li className={index < processingStep ? "complete" : index === processingStep ? "current" : ""} key={step}>
            <span>{index < processingStep ? <Icon name="check" /> : index + 1}</span>
            {step}
          </li>
        ))}
      </ol>
    </div>
  );
}
