import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldSet,
  FieldTitle,
} from '@/components/ui/field'

export function FieldChoiceCard() {
  return (
    <div className="w-full max-w-md">
      <FieldGroup>
        <FieldSet>
          <FieldLabel htmlFor="compute-environment-p8w">
            Compute Environment
          </FieldLabel>
          <FieldDescription>
            Select the compute environment for your cluster.
          </FieldDescription>
          <Field orientation="horizontal">
            <FieldContent>
              <FieldTitle>Kubernetes</FieldTitle>
              <FieldDescription>
                Run GPU workloads on a K8s configured cluster.
              </FieldDescription>
            </FieldContent>
          </Field>
        </FieldSet>
      </FieldGroup>
    </div>
  )
}
